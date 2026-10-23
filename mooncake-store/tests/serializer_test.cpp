#include <glog/logging.h>
#include <gtest/gtest.h>

#include "serializer.h"
#include "serialize/serializer.h"
#include "segment.h"

namespace mooncake::test {

// The pre-placement RPC layout, including the previous compatible addition.
struct LegacyReplicateConfig {
    size_t replica_num{1};
    size_t nof_replica_num{0};
    size_t dfs_replica_num{0};
    SoftPinAction soft_pin_action{SoftPinAction::PRESERVE};
    std::optional<uint64_t> soft_pin_ttl_ms{};
    bool with_hard_pin{false};
    std::vector<std::string> preferred_segments{};
    std::string preferred_segment{};
    std::vector<std::string> preferred_nof_segments{};
    bool prefer_alloc_in_same_node{false};
    ObjectDataType data_type{ObjectDataType::UNKNOWN};
    std::string host_id{};
    std::optional<std::vector<std::string>> group_ids{};
    struct_pack::compatible<std::vector<std::string>, 1> session_ids;
};

struct LegacySegment {
    UUID id{0, 0};
    std::string name{};
    uintptr_t base{0};
    size_t size{0};
    std::string te_endpoint{};
    std::string protocol;
    std::string host_id{};
};

// Example class implementing serialization following the usage documentation
class ExampleClass {
   public:
    ExampleClass() : id_(0), value_(0.0), name_("") {}

    ExampleClass(int id, double value, const std::string& name)
        : id_(id), value_(value), name_(name) {}

    // Serialization method (works with both counter and writer)
    template <typename T>
    void serialize_to(T& serializer) const {
        serializer.write(&id_, sizeof(id_));
        serializer.write(&value_, sizeof(value_));

        // Serialize string length first, then string data
        size_t name_length = name_.length();
        serializer.write(&name_length, sizeof(name_length));
        if (!name_.empty()) {
            serializer.write(name_.data(), name_.length());
        }
    }

    // Deserialization method
    template <typename T>
    static std::shared_ptr<ExampleClass> deserialize_from(T& serializer) {
        try {
            auto obj = std::make_shared<ExampleClass>();

            // Deserialize basic members
            serializer.read(&obj->id_, sizeof(obj->id_));
            serializer.read(&obj->value_, sizeof(obj->value_));

            // Deserialize string
            size_t name_length;
            serializer.read(&name_length, sizeof(name_length));
            if (name_length > 0) {
                obj->name_.resize(name_length);
                serializer.read(&obj->name_[0], name_length);
            }

            return obj;
        } catch (const std::exception& e) {
            return nullptr;
        }
    }

    // Getters for testing
    int getId() const { return id_; }
    double getValue() const { return value_; }
    const std::string& getName() const { return name_; }

    // Equality operator for testing
    bool operator==(const ExampleClass& other) const {
        return id_ == other.id_ && value_ == other.value_ &&
               name_ == other.name_;
    }

   protected:
    int id_;
    double value_;
    std::string name_;
};

class ExampleClassWithException {
   public:
    ExampleClassWithException() : value_(0) {}
    ExampleClassWithException(int value) : value_(value) {}

    template <typename T>
    void serialize_to(T& serializer) const {
        serializer.write(&value_, sizeof(value_));
    }

    template <typename T>
    static std::shared_ptr<ExampleClassWithException> deserialize_from(
        T& serializer) {
        throw std::runtime_error("throw_exception");
    }

   private:
    int value_;
};

class SerializerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("SerializerTest");
        FLAGS_logtostderr = true;
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }
};

TEST_F(SerializerTest, ExampleClassSerialization) {
    // Create an example object
    ExampleClass original(42, 3.14159, "Test Object");

    // Test serialization
    std::vector<SerializedByte> buffer;
    ASSERT_EQ(serialize_to(original, buffer), ErrorCode::OK);
    ASSERT_FALSE(buffer.empty());

    // Test deserialization
    auto restored = deserialize_from<ExampleClass>(buffer);
    ASSERT_NE(restored, nullptr);

    // Verify the deserialized object matches the original
    EXPECT_EQ(restored->getId(), original.getId());
    EXPECT_DOUBLE_EQ(restored->getValue(), original.getValue());
    EXPECT_EQ(restored->getName(), original.getName());
    EXPECT_TRUE(*restored == original);
}

TEST_F(SerializerTest, ExampleClassSerializationWithSharedPtr) {
    // Test with shared_ptr
    auto original =
        std::make_shared<ExampleClass>(777, 2.718, "Shared Pointer Test");

    std::vector<SerializedByte> buffer;
    ASSERT_EQ(serialize_to(original, buffer), ErrorCode::OK);

    auto restored = deserialize_from<ExampleClass>(buffer);
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(*restored == *original);
}

TEST_F(SerializerTest, ExampleClassSerializationNullPointer) {
    // Test with null shared_ptr
    std::shared_ptr<ExampleClass> null_ptr = nullptr;

    std::vector<SerializedByte> buffer;
    ASSERT_EQ(serialize_to(null_ptr, buffer), ErrorCode::INVALID_PARAMS);
}

TEST_F(SerializerTest, ExampleClassDeserializationCorruptedBuffer) {
    // Create a valid object and serialize it
    ExampleClass original(1, 1.0, "Test");
    std::vector<SerializedByte> buffer;
    ASSERT_EQ(serialize_to(original, buffer), ErrorCode::OK);

    // Corrupt the buffer by removing the last byte
    buffer.pop_back();

    // Try to deserialize corrupted buffer
    auto restored = deserialize_from<ExampleClass>(buffer);
    EXPECT_EQ(restored, nullptr);
}

TEST_F(SerializerTest, ExampleClassDeserializationWithException) {
    // Create a valid object and serialize it
    ExampleClassWithException original(1);
    std::vector<SerializedByte> buffer;
    ASSERT_EQ(serialize_to(original, buffer), ErrorCode::OK);

    // Try to deserialize the buffer, the deserialization method will throw an
    // exception.
    auto restored = deserialize_from<ExampleClassWithException>(buffer);
    EXPECT_EQ(restored, nullptr);
}

TEST_F(SerializerTest, MountedSegmentSerializationPreservesHostId) {
    MountedSegment original;
    original.segment.id = generate_uuid();
    original.segment.name = "segment_host1";
    original.segment.base = 0x300000000;
    original.segment.size = 1024 * 1024;
    original.segment.te_endpoint = "segment_host1";
    original.segment.host_id = "host1";
    original.status = SegmentStatus::OK;

    msgpack::sbuffer buffer;
    MsgpackPacker packer(&buffer);
    ASSERT_TRUE(
        Serializer<MountedSegment>::serialize(original, packer).has_value());

    auto object_handle = msgpack::unpack(buffer.data(), buffer.size());
    auto restored =
        Serializer<MountedSegment>::deserialize(object_handle.get());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->segment.id, original.segment.id);
    EXPECT_EQ(restored->segment.name, original.segment.name);
    EXPECT_EQ(restored->segment.host_id, original.segment.host_id);
    EXPECT_EQ(restored->status, original.status);
}

TEST_F(SerializerTest, MountedSegmentDeserializesLegacyFormatWithoutHostId) {
    const UUID segment_id = generate_uuid();

    msgpack::sbuffer buffer;
    MsgpackPacker packer(&buffer);
    packer.pack_array(8);
    packer.pack(UuidToString(segment_id));
    packer.pack(std::string("legacy_segment"));
    packer.pack(static_cast<uint64_t>(0x300000000));
    packer.pack(static_cast<uint64_t>(1024 * 1024));
    packer.pack(std::string("legacy_segment"));
    packer.pack(static_cast<int16_t>(SegmentStatus::OK));
    packer.pack(false);
    packer.pack_nil();

    auto object_handle = msgpack::unpack(buffer.data(), buffer.size());
    auto restored =
        Serializer<MountedSegment>::deserialize(object_handle.get());
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(restored->segment.id, segment_id);
    EXPECT_EQ(restored->segment.name, "legacy_segment");
    EXPECT_TRUE(restored->segment.host_id.empty());
    EXPECT_EQ(restored->status, SegmentStatus::OK);
}

TEST_F(SerializerTest, PlacementRpcCompatibleInBothDirections) {
    LegacyReplicateConfig legacy;
    legacy.host_id = "host1";
    legacy.session_ids = std::vector<std::string>{"session1"};
    auto decoded = struct_pack::deserialize<ReplicateConfig>(
        struct_pack::serialize(legacy));
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->host_id, "host1");
    EXPECT_EQ(decoded->SessionId(), "session1");
    EXPECT_FALSE(decoded->requester_placement);
    EXPECT_FALSE(decoded->fabric_domain);

    decoded->requester_self_id = "rack1.host1#g3";
    decoded->fabric_domain = "fabric1";
    decoded->requester_placement =
        PhysicalPlacement{"fabric1", "rack1", "host1", 3, 2};
    const auto bytes = struct_pack::serialize(*decoded);
    auto restored = struct_pack::deserialize<ReplicateConfig>(bytes);
    ASSERT_TRUE(restored);
    ASSERT_TRUE(restored->requester_placement);
    EXPECT_EQ(*restored->requester_placement, *decoded->requester_placement);
    EXPECT_EQ(*restored->requester_self_id, "rack1.host1#g3");
    auto old_reader = struct_pack::deserialize<LegacyReplicateConfig>(bytes);
    ASSERT_TRUE(old_reader);
    EXPECT_EQ(old_reader->host_id, "host1");
    ASSERT_TRUE(old_reader->session_ids);
    EXPECT_EQ(old_reader->session_ids->front(), "session1");

    LegacySegment old_segment;
    old_segment.name = "rack1.host1#g3/numa2";
    old_segment.te_endpoint = "10.0.0.1:18060";
    old_segment.protocol = "mnnvl";
    auto segment =
        struct_pack::deserialize<Segment>(struct_pack::serialize(old_segment));
    ASSERT_TRUE(segment);
    EXPECT_FALSE(segment->physical_placement);
    segment->physical_placement = *decoded->requester_placement;
    auto segment_bytes = struct_pack::serialize(*segment);
    auto new_segment = struct_pack::deserialize<Segment>(segment_bytes);
    ASSERT_TRUE(new_segment);
    EXPECT_EQ(*new_segment, *segment);
    auto old_segment_reader =
        struct_pack::deserialize<LegacySegment>(segment_bytes);
    ASSERT_TRUE(old_segment_reader);
    EXPECT_EQ(old_segment_reader->te_endpoint, "10.0.0.1:18060");
}

TEST_F(SerializerTest, MountedSegmentPreservesFabricIdentityAndProtocol) {
    MountedSegment original;
    original.segment.id = generate_uuid();
    original.segment.name = "rack1.host1#g3/numa2";
    original.segment.base = 0x300000000;
    original.segment.size = 64 * 1024 * 1024;
    original.segment.te_endpoint = "10.0.0.1:18060";
    original.segment.protocol = "mnnvl";
    original.segment.host_id = "host1";
    original.segment.physical_placement =
        PhysicalPlacement{"fabric1", "rack1", "host1", 3, 2};
    original.buf_allocator = std::make_shared<OffsetBufferAllocator>(
        original.segment.name, original.segment.base, original.segment.size,
        original.segment.te_endpoint);
    original.status = SegmentStatus::OK;
    msgpack::sbuffer buffer;
    MsgpackPacker packer(&buffer);
    ASSERT_TRUE(Serializer<MountedSegment>::serialize(original, packer));
    auto handle = msgpack::unpack(buffer.data(), buffer.size());
    auto restored = Serializer<MountedSegment>::deserialize(handle.get());
    ASSERT_TRUE(restored);
    EXPECT_EQ(restored->segment, original.segment);
    ASSERT_TRUE(restored->buf_allocator);
    EXPECT_EQ(restored->buf_allocator->getTransportEndpoint(),
              original.segment.te_endpoint);
}

TEST_F(SerializerTest, MountedMnnvlSegmentRejectsMissingPhysicalIdentity) {
    MountedSegment original;
    original.segment.id = generate_uuid();
    original.segment.protocol = "mnnvl";
    msgpack::sbuffer buffer;
    MsgpackPacker packer(&buffer);
    ASSERT_TRUE(Serializer<MountedSegment>::serialize(original, packer));
    auto handle = msgpack::unpack(buffer.data(), buffer.size());
    EXPECT_FALSE(Serializer<MountedSegment>::deserialize(handle.get()));
}

}  // namespace mooncake::test

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
