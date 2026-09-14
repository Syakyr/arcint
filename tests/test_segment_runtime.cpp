// THE SEGMENTED RUNTIME'S DATA MOVEMENT, DEVICE-FREE HALF (src/exec/
// segment_runtime.h). No card, no OpenVINO, no IR: a small hand-built plan, a
// blob whose bytes are a function of their offset, and the assertions that a
// refill puts exactly the right bytes in exactly the right span, that ONE set
// serves every segment by reusing the same spans under different port names,
// and that a half-finished refill is never claimed as resident.
//
// The plan here is SYNTHETIC on purpose (4 layers, 2 segments, 8-byte bodies):
// the real geometry is 48 layers and a 14.06 GiB buffer set, and a unit test
// that allocated one would be a benchmark with a assert() stapled to it. The
// real numbers are measured where they can be: the artifact contract by
// `arcint --inspect-artifact`, the staging on a card (window-051 rows (a)/(b')).
#include "exec/segment_runtime.h"
#include "harness.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace lgc;
using namespace lgc::segplan;

namespace {

constexpr uint64_t kBody   = 8;   // bytes a a synthetic body
constexpr uint64_t kBlobLen = 96;  // 2 segments x 2 layers x 3 kinds x 8 bytes

// The blob's content: byte i is a function of i alone, so a refill that reads
// from the wrong offset cannot accidentally look right.
uint8_t blob_byte(uint64_t i) { return static_cast<uint8_t>((i * 7 + 3) % 251); }

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b(kBlobLen);
    for (uint64_t i = 0; i < kBlobLen; ++i) b[i] = blob_byte(i);
    return b;
}

// A plan the real one's SHAPE of: 48 -> 4 layers, 4 -> 2 segments of 2, three
// kinds a layer, slot = (layer - lo) * 3 + kind, offsets ascending in write
// order. Built directly (not through plan_segments) because what these cells
// test is the data movement, not the manifest checks -- those are
// tests/test_segments.cpp's job and they use the real 48-layer geometry.
Plan tiny_plan(uint64_t body_bytes = kBody) {
    Plan p;
    for (int k = 0; k < 2; ++k) {
        SegmentSpec s;
        s.index    = k;
        s.lo       = k * 2;
        s.hi       = k * 2 + 2;
        s.first    = (k == 0);
        s.last     = (k == 1);
        s.in_width = (k == 0) ? 4 : 16;
        p.segments.push_back(s);
    }
    uint64_t off = 0;
    for (int k = 0; k < 2; ++k) {
        for (int layer = k * 2; layer < k * 2 + 2; ++layer) {
            for (int kind = 0; kind < 3; ++kind) {
                BodySpec b;
                b.name       = "layer" + std::to_string(layer) + "/moe/experts_" + kKinds[kind] +
                               "/weight_u8";
                b.segment    = k;
                b.layer      = layer;
                b.kind_index = kind;
                b.shape      = {static_cast<int64_t>(body_bytes)};
                b.offset     = off;
                b.bytes      = body_bytes;
                off         += body_bytes;
                p.bodies.push_back(std::move(b));
            }
        }
    }
    p.slot_shape        = {std::vector<int64_t>{static_cast<int64_t>(body_bytes)},
                           std::vector<int64_t>{static_cast<int64_t>(body_bytes)},
                           std::vector<int64_t>{static_cast<int64_t>(body_bytes)}};
    p.slots_per_segment = 6;  // 3 * max(hi - lo) = 3 * 2
    return p;
}

// A reader over an in-memory blob, with a length the caller controls: short
// blobs are the case that must refuse, not the case that must not crash.
struct MemBlob {
    std::vector<uint8_t> bytes;
    auto reader() const {
        const std::vector<uint8_t>* p = &bytes;
        return [p](uint64_t offset, uint8_t* dst, size_t n) -> size_t {
            size_t got = 0;
            while (got < n && offset + got < p->size()) {
                dst[got] = (*p)[static_cast<size_t>(offset + got)];
                ++got;
            }
            return got;
        };
    }
};

std::string throw_what(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

std::string temp_blob_path() {
    return std::string("/tmp/arcint-segrt-") + std::to_string(static_cast<long>(::getpid())) +
           ".blob";
}

}  // namespace

TEST(segment_runtime_buffer_set_spans_are_the_plans_prefix_sums_and_disjoint) {
    const Plan plan = tiny_plan();
    BufferSet  set(plan);
    CHECK_EQ(set.slots(), static_cast<size_t>(6));
    CHECK_EQ(set.bytes(), static_cast<uint64_t>(6 * kBody));
    CHECK_EQ(set.bytes(), plan.buffer_set_bytes());
    CHECK(set.base() == set.slot_ptr(0));
    for (size_t s = 0; s < set.slots(); ++s) {
        CHECK_EQ(set.slot_bytes(s), kBody);
        if (s + 1 < set.slots()) CHECK(set.slot_ptr(s + 1) == set.slot_ptr(s) + kBody);
    }
    // An out-of-range slot refuses; it does not read past the arena.
    const std::string why = throw_what([&] { (void)set.slot_ptr(6); });
    CHECK(why.find("6 slots, slot 6") != std::string::npos);
    CHECK(throw_what([&] { (void)set.slot_bytes(99); }).find("slot 99") != std::string::npos);
}

TEST(segment_runtime_refill_lands_exactly_the_blobs_bytes_per_slot) {
    const Plan    plan = tiny_plan();
    BufferSet     set(plan);
    BufferSetState st;
    const MemBlob blob{make_blob()};

    const RefillReport rep = refill_segment(plan, 0, set, st, blob.reader());
    CHECK_EQ(rep.segment, 0);
    CHECK_EQ(rep.ops, static_cast<size_t>(6));
    CHECK_EQ(rep.bytes, static_cast<uint64_t>(6 * kBody));
    CHECK_EQ(st.resident, 0);

    // Segment 0's bodies are the blob's first 48 bytes, in write order, so the
    // arena's byte j must equal the blob's byte j.
    for (uint64_t j = 0; j < 6 * kBody; ++j) {
        CHECK_EQ(set.base()[j], blob_byte(j));
    }
}

TEST(segment_runtime_one_set_refilled_for_the_next_segment_reuses_every_slot) {
    const Plan    plan = tiny_plan();
    BufferSet     set(plan);
    BufferSetState st;
    const MemBlob blob{make_blob()};

    // Slot 0 means layer0's gate in segment 0 and layer2's gate in segment 1 --
    // the same span, a different port name. That is the whole route in one line.
    const auto b0 = bind_list(plan, 0);
    const auto b1 = bind_list(plan, 1);
    CHECK_EQ(b0.front().first, static_cast<size_t>(0));
    CHECK_EQ(b1.front().first, static_cast<size_t>(0));
    CHECK(b0.front().second != b1.front().second);

    refill_segment(plan, 0, set, st, blob.reader());
    const uint8_t slot0_first_before = set.slot_ptr(0)[0];
    refill_segment(plan, 1, set, st, blob.reader());
    CHECK_EQ(st.resident, 1);
    // Segment 1's first body starts at blob offset 48.
    CHECK_EQ(set.slot_ptr(0)[0], blob_byte(6 * kBody));
    CHECK(slot0_first_before != set.slot_ptr(0)[0]);
    for (uint64_t j = 0; j < 6 * kBody; ++j) {
        CHECK_EQ(set.base()[j], blob_byte(6 * kBody + j));
    }
}

TEST(segment_runtime_a_short_blob_refuses_and_the_state_stays_on_what_is_resident) {
    const Plan    plan = tiny_plan();
    BufferSet     set(plan);
    BufferSetState st;
    MemBlob       blob{make_blob()};
    blob.bytes.resize(6 * kBody + 3 * kBody);  // segment 1 is only half present

    refill_segment(plan, 0, set, st, blob.reader());
    CHECK_EQ(st.resident, 0);

    const std::string why = throw_what([&] { refill_segment(plan, 1, set, st, blob.reader()); });
    CHECK(why.find("blob is short") != std::string::npos);
    CHECK(why.find("layer3/moe/experts_gate/weight_u8") != std::string::npos);
    // The state did NOT move: the bytes in memory are still segment 0's, and a
    // forward for segment 1 has to hit the guard rather than read half a set.
    CHECK_EQ(st.resident, 0);
    CHECK(throw_what([&] { st.check_before_infer(1); }).find("holds segment 0") !=
          std::string::npos);
}

TEST(segment_runtime_a_plan_that_disagrees_with_the_set_refuses_before_any_read) {
    const Plan     plan8 = tiny_plan(kBody);
    const Plan     plan9 = tiny_plan(kBody + 1);
    BufferSet      set(plan8);
    BufferSetState st;
    const MemBlob  blob{make_blob()};

    const std::string why = throw_what([&] { refill_segment(plan9, 0, set, st, blob.reader()); });
    CHECK(why.find("not the same plan") != std::string::npos);
    CHECK(why.find("slot 0") != std::string::npos);
    CHECK_EQ(st.resident, -1);  // nothing was ever claimed
}

TEST(segment_runtime_bindings_carry_the_port_name_and_the_span_it_must_be_bound_to) {
    const Plan    plan = tiny_plan();
    BufferSet     set(plan);
    const MemBlob blob{make_blob()};

    const auto binds = bindings_for(plan, 1, set);
    CHECK_EQ(binds.size(), static_cast<size_t>(6));
    const auto names = bind_list(plan, 1);
    for (size_t i = 0; i < binds.size(); ++i) {
        CHECK_EQ(binds[i].slot, names[i].first);            // slot order, same as bind_list
        CHECK_EQ(binds[i].name, names[i].second);
        CHECK_EQ(binds[i].dst, set.slot_ptr(binds[i].slot));
        CHECK_EQ(binds[i].bytes, set.slot_bytes(binds[i].slot));
        CHECK(binds[i].dst >= set.base() && binds[i].dst + binds[i].bytes <=
                                                  set.base() + set.bytes());
    }
    CHECK_EQ(binds.front().name, std::string("layer2/moe/experts_gate/weight_u8"));
}

TEST(segment_runtime_a_real_blob_through_a_file_descriptor_refills_and_refuses_short) {
    const std::string path = temp_blob_path();
    const std::vector<uint8_t> whole = make_blob();
    auto write_file = [&](const std::vector<uint8_t>& data) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        CHECK(fd >= 0);
        size_t off = 0;
        while (off < data.size()) {
            const ssize_t w = ::write(fd, data.data() + off, data.size() - off);
            CHECK(w > 0);
            off += static_cast<size_t>(w);
        }
        ::close(fd);
    };

    const Plan     plan = tiny_plan();
    BufferSet      set(plan);
    BufferSetState st;

    write_file(whole);
    {
        FileBlobReader reader = FileBlobReader::open(path);
        const RefillReport rep = refill_segment(plan, 1, set, st, reader);
        CHECK_EQ(rep.ops, static_cast<size_t>(6));
        CHECK_EQ(st.resident, 1);
        for (uint64_t j = 0; j < 6 * kBody; ++j) CHECK_EQ(set.base()[j], blob_byte(48 + j));
    }

    // Now the same reader over a blob that ends mid-segment: the refusal names
    // the body, and the state stays on segment 1 (what is really in memory).
    write_file(std::vector<uint8_t>(whole.begin(), whole.begin() + 48 + 3 * kBody));
    {
        FileBlobReader reader = FileBlobReader::open(path);
        const std::string why =
            throw_what([&] { refill_segment(plan, 1, set, st, reader); });
        CHECK(why.find("blob is short") != std::string::npos);
        CHECK(why.find("layer3/moe/experts_gate/weight_u8") != std::string::npos);
        CHECK_EQ(st.resident, 1);
        // A read that starts past the end returns nothing rather than throwing:
        // "the blob ends here" is information, not an error.
        uint8_t probe = 0;
        CHECK_EQ(reader(4096, &probe, 4), static_cast<size_t>(0));
    }
    ::unlink(path.c_str());
}
