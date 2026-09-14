#pragma once

// THE SEGMENTED RUNTIME'S DATA MOVEMENT, DEVICE-FREE HALF (window-051 §2).
//
// `exec/segment_plan.h` decides WHAT a chain needs: the slot a body sits in,
// the offset it lives at in `expert_bodies.u8`, the shape that slot must have,
// and the refusals that keep a lying manifest from being driven. This header is
// the other half of the same coin and still opens no device: ONE host buffer
// set, the bytes moved into it a segment at a time, the port names a compiled
// segment's request must be pointed at, and the state that says whether a
// forward may start. The OpenVINO side (backend_ov.cpp) wraps `BufferSet`'s
// spans in RemoteTensors and calls `set_tensor`; every decision THAT needs is
// here, and none of it needs a card to be correct.
//
// Two properties this file is responsible for, both pinned by cells:
//
//   * ONE buffer set serves every segment. Slot `s` of segment k and slot `s`
//     of segment k+1 are the same span of the same allocation, bound to port
//     names that differ only by their layer index (segment_plan.h's
//     `body_slot`). That is legal only because a kind's shape is identical in
//     every layer and segment, which plan_segments already refuses otherwise.
//   * the set's state moves ONLY on a complete refill. A failed or short refill
//     leaves the state on the segment whose bytes are actually in memory, so a
//     forward that starts anyway hits `check_before_infer`, not a half-written
//     set. The first cut is serial (refill, then infer), so this is the whole
//     race story in 0.5.1; a double-buffered variant is where the racing refill
//     would come from (window-050 §4.10's route (a) note, window-051 §2).

#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "exec/segment_plan.h"
#include "util/log.h"

namespace lgc::segplan {

// The one host allocation every segment's expert bodies are refilled into.
// Slot spans are prefix sums of the plan's per-slot byte counts (a kind's
// shape, `Plan::buffer_bytes`), so a slot's address is `base + offsets_[slot]`
// and the whole set is `Plan::buffer_set_bytes` long.
//
// Allocation is `std::vector<uint8_t>`, i.e. the process's own heap: the plugin
// is handed a pointer and a size, exactly as the n-gram table ports are today
// (window-050 §4.8). Where that memory should COME FROM (plain heap, a pinned
// USM allocation, a huge-page pool) is the backend's decision and a measured
// one; this class does not pretend to make it.
class BufferSet {
public:
    explicit BufferSet(const Plan& plan) : plan_(plan) {
        offsets_.assign(plan.slots_per_segment + 1, 0);
        for (size_t s = 0; s < plan.slots_per_segment; ++s) {
            offsets_[s + 1] = offsets_[s] + plan.buffer_bytes(s);
        }
        arena_.resize(static_cast<size_t>(offsets_[plan.slots_per_segment]));
    }

    uint8_t* slot_ptr(size_t slot) {
        checked(slot);
        return arena_.data() + offsets_[slot];
    }
    const uint8_t* slot_ptr(size_t slot) const {
        checked(slot);
        return arena_.data() + offsets_[slot];
    }
    uint64_t slot_bytes(size_t slot) const {
        checked(slot);
        return offsets_[slot + 1] - offsets_[slot];
    }

    uint8_t* base() { return arena_.data(); }
    size_t   slots() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }
    uint64_t bytes() const { return offsets_.empty() ? 0 : offsets_.back(); }
    const Plan& plan() const { return plan_; }

private:
    void checked(size_t slot) const {
        const size_t n = offsets_.empty() ? 0 : offsets_.size() - 1;
        if (slot >= n) {
            throw std::runtime_error(log::format(
                "buffer set has %zu slots, slot %zu was asked for", n, slot));
        }
    }

    Plan                      plan_;   // by value: the set outlives the local plan
    std::vector<uint64_t>     offsets_;
    std::vector<uint8_t>      arena_;
};

// One port of a compiled segment: the name the IR declares, and the span of
// the buffer set that must be resident under it while that segment runs.
struct PortBinding {
    std::string name;
    size_t      slot = 0;
    uint8_t*    dst = nullptr;
    uint64_t    bytes = 0;
};

// Every body-port of `segment`, in slot order (the order a driver binds them,
// and the order a caller can compare against `bind_list`).
inline std::vector<PortBinding> bindings_for(const Plan& plan, int segment, BufferSet& set) {
    std::vector<PortBinding> out;
    for (const auto& [slot, name] : bind_list(plan, segment)) {
        PortBinding b;
        b.name  = name;
        b.slot  = slot;
        b.dst   = set.slot_ptr(slot);
        b.bytes = set.slot_bytes(slot);
        out.push_back(std::move(b));
    }
    return out;
}

// The SHAPE of a blob reader: read `n` bytes at blob offset `offset` into
// `dst`, return the number READ (a short read is the caller's problem to name,
// not this type's to hide). Kept as documentation of the contract; the refill
// below is a template over it instead of a std::function, because a reader that
// owns a file descriptor is move-only and must not be forced through a type
// that requires a copy (measured, not theorised: the first build of these cells
// died on libstdc++'s "std::function target must be copy-constructible").
using BlobReader = std::function<size_t(uint64_t offset, uint8_t* dst, size_t n)>;

// A `BlobReader` over an open file. `pread` in a loop, because a short read is
// normal on a large request (and EINTR is normal on any of them); it returns
// what it managed to read, so the caller can tell "the blob ends here" from
// "the disk hiccupped" by the offset it asked for.
class FileBlobReader {
public:
    explicit FileBlobReader(int fd) : fd_(fd) {}

    // Opens for reading; throws with the path and errno on failure.
    static FileBlobReader open(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(log::format("cannot open expert-body blob '%s': %s",
                                                 path.c_str(), std::strerror(errno)));
        }
        return FileBlobReader(fd);
    }
    ~FileBlobReader() {
        if (fd_ >= 0) ::close(fd_);
    }
    FileBlobReader(FileBlobReader&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    FileBlobReader& operator=(FileBlobReader&& o) noexcept {
        if (this != &o) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = o.fd_;
            o.fd_ = -1;
        }
        return *this;
    }
    FileBlobReader(const FileBlobReader&) = delete;
    FileBlobReader& operator=(const FileBlobReader&) = delete;

    size_t operator()(uint64_t offset, uint8_t* dst, size_t n) const {
        size_t got = 0;
        while (got < n) {
            const ssize_t r = ::pread(fd_, dst + got, n - got,
                                      static_cast<off_t>(offset + got));
            if (r == 0) break;  // the blob ends here
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(log::format(
                    "pread of the expert-body blob at offset %llu: %s",
                    static_cast<unsigned long long>(offset + got), std::strerror(errno)));
            }
            got += static_cast<size_t>(r);
        }
        return got;
    }

private:
    int fd_ = -1;
};

struct RefillReport {
    int      segment = -1;
    size_t   ops = 0;        // bodies copied
    uint64_t bytes = 0;
};

// Fill the buffer set with `segment`'s bodies, reading the blob through
// `read`. `state` is the same BufferSetState the driver checks before every
// infer: it moves to `segment` ONLY after the last byte has landed.
//
// Throws (leaving `state` where it was) when the plan and the set disagree
// about a slot's size -- a buffer set built from a different plan or a
// re-exported manifest -- or when a body cannot be read whole. Both refusals
// name the slot and the body, because the operator's next question is which
// segment to re-stage.
template <class Reader>
inline RefillReport refill_segment(const Plan& plan, int segment, BufferSet& set,
                                   BufferSetState& state, const Reader& read) {
    RefillReport rep;
    rep.segment = segment;
    for (const RefillOp& op : refill_ops(plan, segment)) {
        const std::string name = [&] {
            for (const BodySpec& b : plan.bodies) {
                if (b.segment == segment && b.offset == op.offset) return b.name;
            }
            return std::string("<unknown body>");
        }();

        if (op.bytes != set.slot_bytes(op.slot)) {
            throw std::runtime_error(log::format(
                "refill of segment %d: body '%s' is %llu bytes, slot %zu of this buffer set "
                "is %llu -- the plan and the set are not the same plan",
                segment, name.c_str(), static_cast<unsigned long long>(op.bytes), op.slot,
                static_cast<unsigned long long>(set.slot_bytes(op.slot))));
        }
        const size_t got = read(op.offset, set.slot_ptr(op.slot), static_cast<size_t>(op.bytes));
        if (got != op.bytes) {
            throw std::runtime_error(log::format(
                "refill of segment %d read %zu of %llu bytes of body '%s' at blob offset %llu "
                "-- the blob is short of its own index",
                segment, got, static_cast<unsigned long long>(op.bytes), name.c_str(),
                static_cast<unsigned long long>(op.offset)));
        }
        rep.ops++;
        rep.bytes += op.bytes;
    }
    state.refill(segment);  // ONLY now: a partial set is never claimed as resident
    return rep;
}

}  // namespace lgc::segplan
