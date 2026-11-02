#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kStarsRecordSize = 64;
constexpr std::size_t kTsTrackRecordSize = 40;
constexpr std::uint8_t kStepTraceKind = 10;
constexpr std::uint8_t kTaskFlipKind = 14;
constexpr std::uint16_t kStepStartTag = 60000;
constexpr std::uint16_t kStepEndTag = 60001;
constexpr std::uint16_t kMagic = 0x5A5A;
constexpr std::uint16_t kRuntimeLevel = 5000;
constexpr std::uint16_t kHcclLevel = 5500;
constexpr std::uint16_t kNodeLevel = 10000;

struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::uint16_t read_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
           (static_cast<std::uint16_t>(p[1]) << 8U);
}

std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8U) |
           (static_cast<std::uint32_t>(p[2]) << 16U) |
           (static_cast<std::uint32_t>(p[3]) << 24U);
}

std::uint64_t read_u64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(read_u32(p)) |
           (static_cast<std::uint64_t>(read_u32(p + 4)) << 32U);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ParseError("cannot open " + path.string());
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    if (!in.good() && !in.eof()) {
        throw ParseError("cannot read " + path.string());
    }
    return contents.str();
}

std::optional<std::string> regex_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"?([^\\\",}\\s]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    return match[1].str();
}

std::optional<std::int64_t> json_integer(const fs::path& path, const std::string& key) {
    if (!fs::exists(path)) {
        return std::nullopt;
    }
    const auto value = regex_value(read_text_file(path), key);
    if (!value) {
        return std::nullopt;
    }
    try {
        return std::stoll(*value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

struct InputLayout {
    fs::path input;
    fs::path rank_dir;
    fs::path profiler_dir;
    fs::path prof_dir;
    fs::path device_dir;
    fs::path data_dir;
    fs::path host_dir;
    fs::path host_data_dir;
    fs::path info_json;
    fs::path ready_json;
    std::vector<fs::path> stars_files;
    std::vector<fs::path> ts_track_files;
    std::vector<fs::path> task_track_files;
    std::vector<fs::path> api_event_files;
    std::vector<fs::path> hccl_op_info_files;
    std::vector<fs::path> type_dictionary_files;
    std::vector<fs::path> hash_dictionary_files;
};

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::int64_t slice_number(const fs::path& path) {
    static const std::regex pattern(R"(\.slice_([0-9]+)$)");
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_search(name, match, pattern)) {
        return -1;
    }
    return std::stoll(match[1].str());
}

std::vector<fs::path> matching_files(const fs::path& dir, const std::string& prefix) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (starts_with(name, prefix) && slice_number(entry.path()) >= 0) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end(), [](const fs::path& lhs, const fs::path& rhs) {
        const auto lhs_slice = slice_number(lhs);
        const auto rhs_slice = slice_number(rhs);
        return lhs_slice == rhs_slice ? lhs.string() < rhs.string() : lhs_slice < rhs_slice;
    });
    return files;
}

fs::path only_child_with_prefix(const fs::path& dir, const std::string& prefix, bool directory) {
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const bool type_matches = directory ? entry.is_directory() : entry.is_regular_file();
        if (type_matches && starts_with(entry.path().filename().string(), prefix)) {
            matches.push_back(entry.path());
        }
    }
    if (matches.size() != 1) {
        std::ostringstream error;
        error << "expected exactly one " << prefix << "* under " << dir
              << ", found " << matches.size();
        throw ParseError(error.str());
    }
    return matches.front();
}

std::optional<fs::path> optional_child_with_prefix(
        const fs::path& dir, const std::string& prefix, bool directory) {
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const bool type_matches = directory ? entry.is_directory() : entry.is_regular_file();
        if (type_matches && starts_with(entry.path().filename().string(), prefix)) {
            matches.push_back(entry.path());
        }
    }
    if (matches.size() > 1) {
        std::ostringstream error;
        error << "expected at most one " << prefix << "* under " << dir
              << ", found " << matches.size();
        throw ParseError(error.str());
    }
    return matches.empty() ? std::nullopt : std::optional<fs::path>(matches.front());
}

fs::path only_info_json(const fs::path& device_dir) {
    static const std::regex pattern(R"(^info\.json(?:\.[0-9]+)?$)");
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(device_dir)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && std::regex_match(name, pattern)) {
            matches.push_back(entry.path());
        }
    }
    if (matches.size() != 1) {
        std::ostringstream error;
        error << "expected exactly one info.json[.N] under " << device_dir
              << ", found " << matches.size();
        throw ParseError(error.str());
    }
    return matches.front();
}

InputLayout discover_layout(const fs::path& raw_input) {
    const fs::path input = fs::absolute(raw_input).lexically_normal();
    if (!fs::is_directory(input)) {
        throw ParseError("input is not a directory: " + input.string());
    }

    InputLayout layout;
    layout.input = input;
    const std::string input_name = input.filename().string();
    if (starts_with(input_name, "PROF_")) {
        layout.prof_dir = input;
        layout.profiler_dir = input.parent_path();
        layout.rank_dir = layout.profiler_dir.parent_path();
    } else if (input_name == "profiler") {
        layout.profiler_dir = input;
        layout.rank_dir = input.parent_path();
        layout.prof_dir = only_child_with_prefix(input, "PROF_", true);
    } else if (fs::is_directory(input / "profiler")) {
        layout.rank_dir = input;
        layout.profiler_dir = input / "profiler";
        layout.prof_dir = only_child_with_prefix(layout.profiler_dir, "PROF_", true);
    } else {
        throw ParseError("input is not a rank segment, profiler, or PROF_* directory: " + input.string());
    }

    layout.device_dir = only_child_with_prefix(layout.prof_dir, "device_", true);
    layout.data_dir = layout.device_dir / "data";
    if (!fs::is_directory(layout.data_dir)) {
        throw ParseError("missing device data directory: " + layout.data_dir.string());
    }
    layout.info_json = only_info_json(layout.device_dir);
    layout.ready_json = layout.rank_dir / "ready.json";
    layout.stars_files = matching_files(layout.data_dir, "stars_soc.data.");
    layout.ts_track_files = matching_files(layout.data_dir, "ts_track.data.");
    if (layout.stars_files.empty()) {
        throw ParseError("no stars_soc data files under " + layout.data_dir.string());
    }
    // Some valid captures (for example a manually driven Cell smoke test) do
    // not emit framework step trace records.  Device/kernel/HCCL data remains
    // useful, so represent step coverage as unavailable instead of failing the
    // whole rank.  Malformed non-empty ts_track input still fails closed below.

    const auto host_dir = optional_child_with_prefix(layout.prof_dir, "host", true);
    if (host_dir && fs::is_directory(*host_dir / "data")) {
        layout.host_dir = *host_dir;
        layout.host_data_dir = *host_dir / "data";
        layout.task_track_files = matching_files(layout.host_data_dir, "aging.compact.task_track");
        layout.api_event_files = matching_files(layout.host_data_dir, "aging.api_event.data");
        layout.hccl_op_info_files = matching_files(layout.host_data_dir, "aging.compact.hccl_op_info");
        layout.type_dictionary_files = matching_files(
            layout.host_data_dir, "unaging.additional.type_info_dic");
        layout.hash_dictionary_files = matching_files(
            layout.host_data_dir, "unaging.additional.hash_dic");
    }
    return layout;
}

double read_hwts_frequency_mhz(const fs::path& info_json) {
    const auto value = regex_value(read_text_file(info_json), "hwts_frequency");
    if (!value) {
        throw ParseError("missing hwts_frequency in " + info_json.string());
    }
    try {
        const double frequency = std::stod(*value);
        if (!std::isfinite(frequency) || frequency <= 0.0) {
            throw ParseError("invalid hwts_frequency in " + info_json.string());
        }
        return frequency;
    } catch (const std::invalid_argument&) {
        throw ParseError("invalid hwts_frequency in " + info_json.string());
    } catch (const std::out_of_range&) {
        throw ParseError("out-of-range hwts_frequency in " + info_json.string());
    }
}

struct RecordReadResult {
    std::uint64_t bytes = 0;
    std::uint64_t records = 0;
};

RecordReadResult read_fixed_records(
        const std::vector<fs::path>& files,
        std::size_t record_size,
        const std::function<void(const std::uint8_t*)>& consume) {
    constexpr std::size_t kBufferSize = 1U << 20U;
    std::vector<std::uint8_t> buffer(kBufferSize + record_size);
    std::size_t carry = 0;
    RecordReadResult result;

    for (const auto& path : files) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw ParseError("cannot open " + path.string());
        }
        while (in) {
            in.read(reinterpret_cast<char*>(buffer.data() + carry),
                    static_cast<std::streamsize>(kBufferSize));
            const auto got = in.gcount();
            if (got < 0) {
                throw ParseError("cannot read " + path.string());
            }
            result.bytes += static_cast<std::uint64_t>(got);
            const std::size_t available = carry + static_cast<std::size_t>(got);
            const std::size_t complete = available / record_size;
            for (std::size_t i = 0; i < complete; ++i) {
                consume(buffer.data() + i * record_size);
            }
            result.records += complete;
            carry = available % record_size;
            if (carry != 0) {
                std::memmove(buffer.data(), buffer.data() + complete * record_size, carry);
            }
        }
        if (!in.eof()) {
            throw ParseError("cannot read " + path.string());
        }
    }
    if (carry != 0) {
        std::ostringstream error;
        error << "combined input size is not divisible by record size " << record_size
              << "; trailing bytes=" << carry;
        throw ParseError(error.str());
    }
    return result;
}

struct StepEvent {
    std::uint64_t timestamp = 0;
    std::uint64_t index_id = 0;
    std::uint64_t model_id = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t tag_id = 0;
};

struct StepRange {
    std::uint64_t index_id = 0;
    std::uint64_t model_id = 0;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct DeviceFlip {
    std::uint64_t timestamp = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t flip_num = 0;
};

struct StepParseResult {
    RecordReadResult input;
    std::uint64_t step_trace_records = 0;
    std::uint64_t other_records = 0;
    std::uint64_t unmatched_starts = 0;
    std::uint64_t unmatched_ends = 0;
    std::vector<StepRange> steps;
    std::vector<DeviceFlip> task_flips;
};

struct StepKey {
    std::uint64_t index_id;
    std::uint64_t model_id;

    bool operator<(const StepKey& other) const {
        return std::tie(index_id, model_id) < std::tie(other.index_id, other.model_id);
    }
};

StepParseResult parse_steps(const std::vector<fs::path>& files) {
    StepParseResult result;
    std::vector<StepEvent> events;
    result.input = read_fixed_records(files, kTsTrackRecordSize, [&](const std::uint8_t* p) {
        if (p[1] == kTaskFlipKind) {
            result.task_flips.push_back({
                read_u64(p + 8), read_u16(p + 16), read_u16(p + 22), read_u16(p + 18)
            });
            ++result.other_records;
            return;
        }
        if (p[1] != kStepTraceKind) {
            ++result.other_records;
            return;
        }
        ++result.step_trace_records;
        StepEvent event;
        event.timestamp = read_u64(p + 8);
        event.index_id = read_u64(p + 16);
        event.model_id = read_u64(p + 24);
        event.stream_id = read_u16(p + 32) % (1U << 11U);
        event.task_id = read_u16(p + 34);
        event.tag_id = read_u16(p + 36);
        if (event.tag_id == kStepStartTag || event.tag_id == kStepEndTag) {
            events.push_back(event);
        }
    });

    std::stable_sort(events.begin(), events.end(), [](const StepEvent& lhs, const StepEvent& rhs) {
        return lhs.timestamp < rhs.timestamp;
    });
    std::map<StepKey, std::deque<StepEvent>> starts;
    for (const auto& event : events) {
        const StepKey key{event.index_id, event.model_id};
        if (event.tag_id == kStepStartTag) {
            starts[key].push_back(event);
            continue;
        }
        auto found = starts.find(key);
        if (found == starts.end() || found->second.empty()) {
            ++result.unmatched_ends;
            continue;
        }
        const auto start = found->second.front();
        found->second.pop_front();
        if (event.timestamp < start.timestamp) {
            ++result.unmatched_ends;
            ++result.unmatched_starts;
            continue;
        }
        result.steps.push_back({event.index_id, event.model_id, start.timestamp, event.timestamp});
    }
    for (const auto& item : starts) {
        result.unmatched_starts += item.second.size();
    }
    std::sort(result.steps.begin(), result.steps.end(), [](const StepRange& lhs, const StepRange& rhs) {
        return lhs.start < rhs.start;
    });
    return result;
}

std::uint16_t normalize_stream_id(std::uint16_t stream_id, std::uint16_t task_id) {
    if ((stream_id & 0x1000U) != 0) {
        return stream_id % (1U << 11U);
    }
    if ((stream_id & 0x2000U) != 0) {
        stream_id = task_id & 0x0FFFU;
    }
    return stream_id % (1U << 11U);
}

std::uint16_t normalize_task_id(std::uint16_t stream_id, std::uint16_t task_id) {
    if ((stream_id & 0x1000U) != 0) {
        task_id &= 0x1FFFU;
        task_id |= stream_id & 0xE000U;
    } else if ((stream_id & 0x2000U) != 0) {
        task_id = (stream_id & 0x0FFFU) | (task_id & 0xF000U);
    }
    return task_id;
}

struct StarEvent {
    std::uint64_t syscnt = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint8_t acc_id = 0;
    std::uint8_t task_type = 0;
    bool start = false;
};

struct MatchedTask {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint8_t acc_id = 0;
    std::uint8_t task_type = 0;
    std::uint16_t batch_id = 0;
    std::int64_t step_index = -1;

    std::uint64_t duration_ticks() const { return end - start; }
};

struct StarsParseResult {
    RecordReadResult input;
    std::uint64_t acsq_starts = 0;
    std::uint64_t acsq_ends = 0;
    std::uint64_t other_records = 0;
    std::uint64_t unmatched_starts = 0;
    std::uint64_t unmatched_ends = 0;
    std::vector<MatchedTask> tasks;
};

std::uint64_t task_key(std::uint16_t stream_id, std::uint16_t task_id) {
    return (static_cast<std::uint64_t>(stream_id) << 32U) | task_id;
}

StarsParseResult parse_stars(const std::vector<fs::path>& files) {
    StarsParseResult result;
    std::vector<StarEvent> events;
    std::uint64_t total_bytes = 0;
    for (const auto& file : files) {
        total_bytes += fs::file_size(file);
    }
    events.reserve(static_cast<std::size_t>(total_bytes / (2 * kStarsRecordSize)));

    result.input = read_fixed_records(files, kStarsRecordSize, [&](const std::uint8_t* p) {
        const std::uint16_t header = read_u16(p);
        const std::uint8_t func_type = static_cast<std::uint8_t>(header & 0x3FU);
        if (func_type != 0 && func_type != 1) {
            ++result.other_records;
            return;
        }
        const std::uint16_t raw_stream = read_u16(p + 4);
        const std::uint16_t raw_task = read_u16(p + 6);
        StarEvent event;
        event.syscnt = read_u64(p + 8);
        event.stream_id = normalize_stream_id(raw_stream, raw_task);
        event.task_id = normalize_task_id(raw_stream, raw_task);
        event.acc_id = static_cast<std::uint8_t>(read_u16(p + 18) & 63U);
        event.task_type = static_cast<std::uint8_t>(header >> 10U);
        event.start = func_type == 0;
        events.push_back(event);
        if (event.start) {
            ++result.acsq_starts;
        } else {
            ++result.acsq_ends;
        }
    });

    std::stable_sort(events.begin(), events.end(), [](const StarEvent& lhs, const StarEvent& rhs) {
        return lhs.syscnt < rhs.syscnt;
    });
    std::unordered_map<std::uint64_t, std::deque<StarEvent>> starts;
    starts.reserve(events.size() / 8);
    result.tasks.reserve(std::min(result.acsq_starts, result.acsq_ends));
    for (const auto& event : events) {
        const auto key = task_key(event.stream_id, event.task_id);
        if (event.start) {
            starts[key].push_back(event);
            continue;
        }
        auto found = starts.find(key);
        if (found == starts.end() || found->second.empty()) {
            ++result.unmatched_ends;
            continue;
        }
        const auto start = found->second.front();
        found->second.pop_front();
        if (event.syscnt < start.syscnt) {
            ++result.unmatched_starts;
            ++result.unmatched_ends;
            continue;
        }
        result.tasks.push_back({start.syscnt, event.syscnt, start.stream_id, start.task_id,
                                start.acc_id, start.task_type, 0, -1});
    }
    for (const auto& item : starts) {
        result.unmatched_starts += item.second.size();
    }
    std::sort(result.tasks.begin(), result.tasks.end(), [](const MatchedTask& lhs, const MatchedTask& rhs) {
        return lhs.start < rhs.start;
    });
    return result;
}

void assign_device_batches(std::vector<MatchedTask>& tasks, const std::vector<DeviceFlip>& flips) {
    std::map<std::uint16_t, std::vector<std::size_t>> task_groups;
    std::map<std::uint16_t, std::vector<DeviceFlip>> flip_groups;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        task_groups[tasks[i].stream_id].push_back(i);
    }
    for (const auto& flip : flips) {
        flip_groups[flip.stream_id].push_back(flip);
    }
    for (auto& group : task_groups) {
        auto& indices = group.second;
        std::sort(indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
            return tasks[lhs].start < tasks[rhs].start;
        });
        auto& stream_flips = flip_groups[group.first];
        std::sort(stream_flips.begin(), stream_flips.end(), [](const DeviceFlip& lhs, const DeviceFlip& rhs) {
            return lhs.timestamp < rhs.timestamp;
        });
        std::size_t flip_index = 0;
        std::size_t task_index = 0;
        while (task_index < indices.size()) {
            auto& task = tasks[indices[task_index]];
            if (flip_index < stream_flips.size() && task.start > stream_flips[flip_index].timestamp) {
                const auto& flip = stream_flips[flip_index];
                ++flip_index;
                if (flip.flip_num != std::numeric_limits<std::uint16_t>::max()) {
                    std::size_t backward = task_index;
                    while (backward > 0) {
                        auto& previous = tasks[indices[backward - 1]];
                        if (previous.task_id >= flip.task_id) {
                            break;
                        }
                        previous.batch_id = static_cast<std::uint16_t>(flip_index);
                        --backward;
                    }
                }
                continue;
            }
            task.batch_id = static_cast<std::uint16_t>(flip_index);
            ++task_index;
        }
    }
}

const char* task_type_name(std::uint8_t type) {
    static constexpr std::array<const char*, 21> names = {
        "AI_CORE", "AI_CPU", "AIV_SQE", "PLACE_HOLDER_SQE", "EVENT_RECORD_SQE",
        "EVENT_WAIT_SQE", "NOTIFY_RECORD_SQE", "NOTIFY_WAIT_SQE", "WRITE_VALUE_SQE",
        "VQ6_SQE", "TOF_SQE", "SDMA_SQE", "VPC_SQE", "JPEGE_SQE", "JPEGD_SQE",
        "DSA_SQE", "ROCCE_SQE", "PCIE_DMA_SQE", "HOST_CPU_SQE", "CDQM_SQE", "C_CORE_SQE"
    };
    return type < names.size() ? names[type] : "UNKNOWN";
}

std::uint64_t type_dictionary_key(std::uint16_t level, std::uint32_t type) {
    return (static_cast<std::uint64_t>(level) << 32U) | type;
}

struct HostDictionaries {
    std::unordered_map<std::uint64_t, std::string> type_names;
    std::unordered_map<std::uint64_t, std::string> item_names;
};

std::string trim_line(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ')) {
        value.pop_back();
    }
    const auto first = value.find_first_not_of(" \t");
    return first == std::string::npos ? std::string() : value.substr(first);
}

std::optional<std::uint64_t> parse_uint64(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        return consumed == value.size() ? std::optional<std::uint64_t>(parsed) : std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

HostDictionaries read_host_dictionaries(const InputLayout& layout) {
    HostDictionaries dictionaries;
    for (const auto& path : layout.type_dictionary_files) {
        std::ifstream input(path);
        if (!input) {
            throw ParseError("cannot open " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            line = trim_line(std::move(line));
            const auto colon = line.find(':');
            const auto underscore = line.find('_');
            if (colon == std::string::npos || underscore == std::string::npos || underscore > colon) {
                continue;
            }
            const auto level = parse_uint64(line.substr(0, underscore));
            const auto type = parse_uint64(line.substr(underscore + 1, colon - underscore - 1));
            if (!level || !type || *level > std::numeric_limits<std::uint16_t>::max() ||
                *type > std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }
            dictionaries.type_names[type_dictionary_key(
                static_cast<std::uint16_t>(*level), static_cast<std::uint32_t>(*type))] = line.substr(colon + 1);
        }
    }
    for (const auto& path : layout.hash_dictionary_files) {
        std::ifstream input(path);
        if (!input) {
            throw ParseError("cannot open " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            line = trim_line(std::move(line));
            const auto colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const auto key = parse_uint64(line.substr(0, colon));
            if (key) {
                dictionaries.item_names[*key] = line.substr(colon + 1);
            }
        }
    }
    return dictionaries;
}

std::string lookup_type_name(
        const HostDictionaries& dictionaries, std::uint16_t level, std::uint32_t type) {
    const auto found = dictionaries.type_names.find(type_dictionary_key(level, type));
    return found == dictionaries.type_names.end() ? std::to_string(type) : found->second;
}

std::string lookup_item_name(const HostDictionaries& dictionaries, std::uint64_t item_id) {
    const auto found = dictionaries.item_names.find(item_id);
    return found == dictionaries.item_names.end() ? std::to_string(item_id) : found->second;
}

struct ApiInterval {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint32_t thread_id = 0;
    std::string struct_type;
    std::string item_name;
};

struct IntervalIndex {
    std::vector<ApiInterval> intervals;
    std::vector<std::uint64_t> prefix_max_end;

    void lock() {
        std::stable_sort(intervals.begin(), intervals.end(), [](const ApiInterval& lhs, const ApiInterval& rhs) {
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.end > rhs.end;
        });
        prefix_max_end.resize(intervals.size());
        std::uint64_t maximum = 0;
        for (std::size_t i = 0; i < intervals.size(); ++i) {
            maximum = std::max(maximum, intervals[i].end);
            prefix_max_end[i] = maximum;
        }
    }

    const ApiInterval* containing(std::uint64_t timestamp) const {
        const auto upper = std::upper_bound(
            intervals.begin(), intervals.end(), timestamp,
            [](std::uint64_t value, const ApiInterval& interval) { return value < interval.start; });
        if (upper == intervals.begin()) {
            return nullptr;
        }
        std::size_t index = static_cast<std::size_t>(std::distance(intervals.begin(), upper) - 1);
        while (true) {
            const auto& candidate = intervals[index];
            if (candidate.start <= timestamp && timestamp <= candidate.end) {
                return &candidate;
            }
            if (index == 0 || prefix_max_end[index - 1] < timestamp) {
                break;
            }
            --index;
        }
        return nullptr;
    }
};

struct HostTrack {
    std::uint64_t timestamp = 0;
    std::uint64_t task_type_id = 0;
    std::uint32_t thread_id = 0;
    std::uint16_t device_id = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t batch_id = 0;
    std::string task_type;
    std::string struct_type;
};

struct HostFlip {
    std::uint64_t timestamp = 0;
    std::uint16_t device_id = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t flip_num = 0;
};

struct HostKernel {
    std::uint64_t host_timestamp = 0;
    std::uint32_t thread_id = 0;
    std::uint16_t device_id = 0;
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t batch_id = 0;
    std::string host_task_type;
    std::string name;
    std::string name_source;
};

struct HcclOp {
    std::uint64_t timestamp = 0;
    std::uint64_t host_start = 0;
    std::uint64_t host_end = 0;
    std::uint64_t element_count = 0;
    std::uint64_t group_hash = 0;
    std::uint32_t thread_id = 0;
    bool relay = false;
    bool retry = false;
    std::string op_name;
    std::string data_type;
    std::string algorithm;
    std::string group_name;
};

struct HostParseResult {
    bool available = false;
    RecordReadResult api_input;
    RecordReadResult task_input;
    RecordReadResult hccl_op_input;
    std::uint64_t regular_api_records = 0;
    std::uint64_t api_event_records = 0;
    std::uint64_t node_intervals = 0;
    std::uint64_t hccl_intervals = 0;
    std::uint64_t normal_task_tracks = 0;
    std::uint64_t flip_task_tracks = 0;
    std::uint64_t maintenance_task_tracks = 0;
    std::uint64_t kernel_task_tracks = 0;
    std::uint64_t kernel_tracks_without_name = 0;
    std::uint64_t hccl_ops_without_node = 0;
    std::vector<HostKernel> kernels;
    std::vector<HcclOp> hccl_ops;
};

const char* hccl_data_type_name(std::uint8_t value) {
    static constexpr std::array<const char*, 13> names = {
        "INT8", "INT16", "INT32", "FP16", "FP32", "INT64", "UINT64",
        "UINT8", "UINT16", "UINT32", "FP64", "BFP16", "INT128"
    };
    return value < names.size() ? names[value] : "UNKNOWN";
}

const char* hccl_algorithm_phase_name(std::uint8_t value) {
    static constexpr std::array<const char*, 9> names = {
        "NONE", "MESH", "RING", "NB", "HD", "NHR", "PIPELINE", "PAIRWISE", "STAR"
    };
    return value < names.size() ? names[value] : "UNKNOWN";
}

std::string hccl_algorithm_name(std::uint16_t encoded) {
    std::ostringstream result;
    bool first = true;
    for (int phase_index = 0; phase_index < 4; ++phase_index) {
        const auto phase = static_cast<std::uint8_t>(encoded & 0xFU);
        if (phase == 0) {
            break;
        }
        if (!first) {
            result << '-';
        }
        result << hccl_algorithm_phase_name(phase);
        first = false;
        encoded >>= 4U;
    }
    return first ? "NONE" : result.str();
}

std::uint32_t host_stream_key(std::uint16_t device_id, std::uint16_t stream_id) {
    return (static_cast<std::uint32_t>(device_id) << 16U) | stream_id;
}

void assign_host_batches(std::vector<HostTrack>& tasks, const std::vector<HostFlip>& flips) {
    std::map<std::uint32_t, std::vector<std::size_t>> task_groups;
    std::map<std::uint32_t, std::vector<HostFlip>> flip_groups;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        task_groups[host_stream_key(tasks[i].device_id, tasks[i].stream_id)].push_back(i);
    }
    for (const auto& flip : flips) {
        flip_groups[host_stream_key(flip.device_id, flip.stream_id)].push_back(flip);
    }
    for (auto& group : task_groups) {
        auto& indices = group.second;
        std::sort(indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
            return tasks[lhs].timestamp < tasks[rhs].timestamp;
        });
        auto& stream_flips = flip_groups[group.first];
        std::sort(stream_flips.begin(), stream_flips.end(), [](const HostFlip& lhs, const HostFlip& rhs) {
            return lhs.timestamp < rhs.timestamp;
        });
        std::size_t flip_index = 0;
        std::size_t task_index = 0;
        while (task_index < indices.size()) {
            auto& task = tasks[indices[task_index]];
            if (flip_index < stream_flips.size() && task.timestamp > stream_flips[flip_index].timestamp) {
                const auto& flip = stream_flips[flip_index];
                ++flip_index;
                if (flip.flip_num != std::numeric_limits<std::uint16_t>::max()) {
                    std::size_t backward = task_index;
                    while (backward > 0) {
                        auto& previous = tasks[indices[backward - 1]];
                        if (previous.task_id >= flip.task_id) {
                            break;
                        }
                        previous.batch_id = static_cast<std::uint16_t>(flip_index);
                        --backward;
                    }
                }
                continue;
            }
            task.batch_id = static_cast<std::uint16_t>(flip_index);
            ++task_index;
        }
    }
}

bool is_kernel_host_task(const std::string& task_type, bool inside_hccl) {
    if (starts_with(task_type, "KERNEL")) {
        return true;
    }
    if (task_type == "STARS_COMMON") {
        return true;
    }
    return task_type == "FFTS_PLUS" && !inside_hccl;
}

HostParseResult parse_host_names(const InputLayout& layout) {
    HostParseResult result;
    if (layout.task_track_files.empty() || layout.api_event_files.empty() ||
        layout.type_dictionary_files.empty() || layout.hash_dictionary_files.empty()) {
        return result;
    }
    result.available = true;
    const HostDictionaries dictionaries = read_host_dictionaries(layout);
    std::unordered_map<std::uint32_t, IntervalIndex> node_indices;
    std::unordered_map<std::uint32_t, IntervalIndex> hccl_indices;

    result.api_input = read_fixed_records(layout.api_event_files, 40, [&](const std::uint8_t* p) {
        if (read_u16(p) != kMagic) {
            throw ParseError("invalid magic in api_event data");
        }
        const std::uint16_t level = read_u16(p + 2);
        const std::uint32_t struct_type_id = read_u32(p + 4);
        const std::uint32_t thread_id = read_u32(p + 8);
        const std::uint64_t start = read_u64(p + 16);
        const std::uint64_t end = read_u64(p + 24);
        const std::uint64_t item_id = read_u64(p + 32);
        if (end == std::numeric_limits<std::uint64_t>::max()) {
            ++result.api_event_records;
            return;
        }
        ++result.regular_api_records;
        if (start == 0 || start == end || end < start) {
            return;
        }
        ApiInterval interval{
            start, end, thread_id,
            lookup_type_name(dictionaries, level, struct_type_id),
            lookup_item_name(dictionaries, item_id)
        };
        if (level == kNodeLevel) {
            node_indices[thread_id].intervals.push_back(std::move(interval));
            ++result.node_intervals;
        } else if (level == kHcclLevel) {
            hccl_indices[thread_id].intervals.push_back(std::move(interval));
            ++result.hccl_intervals;
        }
    });
    for (auto& item : node_indices) {
        item.second.lock();
    }
    for (auto& item : hccl_indices) {
        item.second.lock();
    }

    if (!layout.hccl_op_info_files.empty()) {
        result.hccl_op_input = read_fixed_records(
            layout.hccl_op_info_files, 64, [&](const std::uint8_t* p) {
                if (read_u16(p) != kMagic) {
                    throw ParseError("invalid magic in hccl_op_info data");
                }
                const std::uint32_t thread_id = read_u32(p + 8);
                const std::uint64_t timestamp = read_u64(p + 16);
                const std::uint8_t flags = p[24];
                const std::uint8_t data_type = p[25];
                const std::uint16_t algorithm = read_u16(p + 26);
                const std::uint64_t element_count = read_u64(p + 28);
                const std::uint64_t group_hash = read_u64(p + 36);
                const ApiInterval* node = nullptr;
                const auto node_found = node_indices.find(thread_id);
                if (node_found != node_indices.end()) {
                    node = node_found->second.containing(timestamp);
                }
                if (node == nullptr) {
                    ++result.hccl_ops_without_node;
                }
                result.hccl_ops.push_back({
                    timestamp,
                    node == nullptr ? 0 : node->start,
                    node == nullptr ? 0 : node->end,
                    element_count,
                    group_hash,
                    thread_id,
                    (flags & 0x1U) != 0,
                    (flags & 0x2U) != 0,
                    node == nullptr ? "UNRESOLVED" : node->item_name,
                    hccl_data_type_name(data_type),
                    hccl_algorithm_name(algorithm),
                    lookup_item_name(dictionaries, group_hash),
                });
            });
    }

    std::vector<HostTrack> tracks;
    std::vector<HostFlip> flips;
    result.task_input = read_fixed_records(layout.task_track_files, 64, [&](const std::uint8_t* p) {
        if (read_u16(p) != kMagic) {
            throw ParseError("invalid magic in task_track data");
        }
        const std::uint16_t level = read_u16(p + 2);
        const std::uint32_t struct_type_id = read_u32(p + 4);
        const std::uint32_t thread_id = read_u32(p + 8);
        const std::uint64_t timestamp = read_u64(p + 16);
        const std::uint16_t device_id = read_u16(p + 24);
        const std::uint16_t stream_id = read_u16(p + 26);
        const std::uint16_t task_id = read_u16(p + 28);
        const std::uint16_t raw_batch_id = read_u16(p + 30);
        const std::uint64_t task_type_id = read_u64(p + 32);
        const std::string task_type = lookup_type_name(
            dictionaries, level, static_cast<std::uint32_t>(task_type_id));
        if (task_type == "FLIP_TASK") {
            flips.push_back({timestamp, device_id, stream_id, task_id, raw_batch_id});
            ++result.flip_task_tracks;
            return;
        }
        if (task_type == "MAINTENANCE") {
            ++result.maintenance_task_tracks;
            return;
        }
        tracks.push_back({timestamp, task_type_id, thread_id, device_id, stream_id, task_id, 0,
                          task_type, lookup_type_name(dictionaries, level, struct_type_id)});
        ++result.normal_task_tracks;
    });
    assign_host_batches(tracks, flips);

    for (const auto& track : tracks) {
        const ApiInterval* node = nullptr;
        const ApiInterval* hccl = nullptr;
        const auto node_found = node_indices.find(track.thread_id);
        if (node_found != node_indices.end()) {
            node = node_found->second.containing(track.timestamp);
        }
        const auto hccl_found = hccl_indices.find(track.thread_id);
        if (hccl_found != hccl_indices.end()) {
            hccl = hccl_found->second.containing(track.timestamp);
        }
        if (!is_kernel_host_task(track.task_type, hccl != nullptr)) {
            continue;
        }
        ++result.kernel_task_tracks;
        const ApiInterval* name_interval = hccl != nullptr ? hccl : node;
        if (name_interval == nullptr || name_interval->item_name.empty() || name_interval->item_name == "0") {
            ++result.kernel_tracks_without_name;
            continue;
        }
        result.kernels.push_back({
            track.timestamp, track.thread_id, track.device_id, track.stream_id, track.task_id,
            track.batch_id, track.task_type, name_interval->item_name,
            hccl != nullptr ? "hccl_api" : "node_api"
        });
    }
    std::sort(result.kernels.begin(), result.kernels.end(), [](const HostKernel& lhs, const HostKernel& rhs) {
        return lhs.host_timestamp < rhs.host_timestamp;
    });
    return result;
}

double ticks_to_ns(double ticks, double frequency_mhz) {
    return ticks * 1000.0 / frequency_mhz;
}

std::int64_t assign_step(const std::vector<StepRange>& steps, std::uint64_t timestamp) {
    const auto found = std::upper_bound(steps.begin(), steps.end(), timestamp,
        [](std::uint64_t time, const StepRange& step) { return time < step.start; });
    if (found == steps.begin()) {
        return -1;
    }
    const auto& candidate = *std::prev(found);
    if (timestamp >= candidate.start && timestamp <= candidate.end) {
        return static_cast<std::int64_t>(candidate.index_id);
    }
    return -1;
}

double quantile_sorted(const std::vector<std::uint64_t>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double position = q * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) {
        return static_cast<double>(sorted[lower]);
    }
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - fraction) +
           static_cast<double>(sorted[upper]) * fraction;
}

struct TaskTypeStats {
    std::uint8_t type = 0;
    std::uint64_t total_ticks = 0;
    std::vector<std::uint64_t> durations;
};

struct TaskIdentity {
    std::uint16_t stream_id = 0;
    std::uint16_t task_id = 0;
    std::uint16_t batch_id = 0;

    bool operator==(const TaskIdentity& other) const {
        return stream_id == other.stream_id && task_id == other.task_id && batch_id == other.batch_id;
    }
};

struct TaskIdentityHash {
    std::size_t operator()(const TaskIdentity& value) const {
        const std::uint64_t packed = (static_cast<std::uint64_t>(value.stream_id) << 32U) |
                                     (static_cast<std::uint64_t>(value.task_id) << 16U) |
                                     value.batch_id;
        return std::hash<std::uint64_t>{}(packed);
    }
};

struct NamedKernelStats {
    std::string name;
    std::string name_source;
    std::string host_task_type;
    std::uint16_t stream_id = 0;
    std::uint8_t device_task_type = 0;
    std::uint64_t total_ticks = 0;
    std::vector<std::uint64_t> durations;
};

struct HcclOpStats {
    std::string op_name;
    std::string data_type;
    std::string algorithm;
    std::string group_name;
    std::uint64_t group_hash = 0;
    std::uint64_t element_count = 0;
    bool relay = false;
    bool retry = false;
    std::vector<std::uint64_t> host_durations;
};

std::string render_json(const InputLayout& layout,
                        double frequency_mhz,
                        const StepParseResult& step_result,
                        StarsParseResult& stars_result,
                        const HostParseResult& host_result,
                        double parse_seconds,
                        std::size_t top_k) {
    std::uint64_t assigned_tasks = 0;
    std::map<std::uint8_t, TaskTypeStats> per_type;
    for (auto& task : stars_result.tasks) {
        task.step_index = assign_step(step_result.steps, task.start);
        if (task.step_index >= 0) {
            ++assigned_tasks;
        }
        auto& stats = per_type[task.task_type];
        stats.type = task.task_type;
        stats.total_ticks += task.duration_ticks();
        stats.durations.push_back(task.duration_ticks());
    }
    for (auto& item : per_type) {
        std::sort(item.second.durations.begin(), item.second.durations.end());
    }

    std::unordered_map<TaskIdentity, const HostKernel*, TaskIdentityHash> host_kernel_by_task;
    host_kernel_by_task.reserve(host_result.kernels.size());
    std::uint64_t duplicate_host_kernel_keys = 0;
    for (const auto& kernel : host_result.kernels) {
        const TaskIdentity identity{kernel.stream_id, kernel.task_id, kernel.batch_id};
        const auto inserted = host_kernel_by_task.emplace(identity, &kernel);
        if (!inserted.second) {
            ++duplicate_host_kernel_keys;
        }
    }

    using NamedKernelKey = std::tuple<std::string, std::uint16_t, std::uint8_t, std::string, std::string>;
    std::map<NamedKernelKey, NamedKernelStats> named_kernel_stats;
    std::uint64_t named_device_tasks = 0;
    std::uint64_t eligible_compute_tasks = 0;
    std::uint64_t named_compute_tasks = 0;
    for (const auto& task : stars_result.tasks) {
        const bool eligible_compute = task.task_type == 0 || task.task_type == 1;
        if (eligible_compute) {
            ++eligible_compute_tasks;
        }
        const auto found = host_kernel_by_task.find({task.stream_id, task.task_id, task.batch_id});
        if (found == host_kernel_by_task.end()) {
            continue;
        }
        ++named_device_tasks;
        if (eligible_compute) {
            ++named_compute_tasks;
        }
        const auto& kernel = *found->second;
        const NamedKernelKey key{
            kernel.name, task.stream_id, task.task_type, kernel.host_task_type, kernel.name_source
        };
        auto& stats = named_kernel_stats[key];
        stats.name = kernel.name;
        stats.name_source = kernel.name_source;
        stats.host_task_type = kernel.host_task_type;
        stats.stream_id = task.stream_id;
        stats.device_task_type = task.task_type;
        stats.total_ticks += task.duration_ticks();
        stats.durations.push_back(task.duration_ticks());
    }
    for (auto& item : named_kernel_stats) {
        std::sort(item.second.durations.begin(), item.second.durations.end());
    }
    const std::uint64_t unmatched_host_kernels = host_result.kernels.size() >= named_device_tasks ?
        host_result.kernels.size() - named_device_tasks : 0;

    using HcclKey = std::tuple<
        std::string, std::string, std::string, std::uint64_t, std::uint64_t, bool, bool>;
    std::map<HcclKey, HcclOpStats> hccl_stats;
    for (const auto& hccl : host_result.hccl_ops) {
        const HcclKey key{
            hccl.op_name, hccl.data_type, hccl.algorithm, hccl.group_hash,
            hccl.element_count, hccl.relay, hccl.retry
        };
        auto& stats = hccl_stats[key];
        stats.op_name = hccl.op_name;
        stats.data_type = hccl.data_type;
        stats.algorithm = hccl.algorithm;
        stats.group_name = hccl.group_name;
        stats.group_hash = hccl.group_hash;
        stats.element_count = hccl.element_count;
        stats.relay = hccl.relay;
        stats.retry = hccl.retry;
        if (hccl.host_start != 0 && hccl.host_end >= hccl.host_start) {
            stats.host_durations.push_back(hccl.host_end - hccl.host_start);
        }
    }
    for (auto& item : hccl_stats) {
        std::sort(item.second.host_durations.begin(), item.second.host_durations.end());
    }

    std::vector<const MatchedTask*> slowest;
    slowest.reserve(stars_result.tasks.size());
    for (const auto& task : stars_result.tasks) {
        slowest.push_back(&task);
    }
    const std::size_t keep = std::min(top_k, slowest.size());
    std::partial_sort(slowest.begin(), slowest.begin() + keep, slowest.end(),
        [](const MatchedTask* lhs, const MatchedTask* rhs) {
            return lhs->duration_ticks() > rhs->duration_ticks();
        });
    slowest.resize(keep);

    const auto rank = json_integer(layout.ready_json, "rank");
    const auto segment = json_integer(layout.ready_json, "segment");
    const auto first_step = json_integer(layout.ready_json, "first_step");
    const auto last_step = json_integer(layout.ready_json, "last_step");
    std::optional<std::uint64_t> expected_step_pairs;
    if (first_step && last_step && *last_step >= *first_step) {
        expected_step_pairs = *last_step - *first_step + 1;
    }
    const bool step_trace_available = !step_result.steps.empty();
    const bool step_trace_complete = expected_step_pairs
        ? step_result.steps.size() == *expected_step_pairs
            && step_result.unmatched_starts == 0 && step_result.unmatched_ends == 0
        : step_trace_available
            && step_result.unmatched_starts == 0 && step_result.unmatched_ends == 0;

    std::ostringstream out;
    out << std::setprecision(15);
    out << "{\n";
    out << "  \"schema\": \"fast_summary_v1\",\n";
    out << "  \"parser\": {\"name\": \"fast_parse\", \"version\": 1, "
        << "\"parse_seconds\": " << parse_seconds << "},\n";
    out << "  \"source\": {\n";
    out << "    \"input\": \"" << json_escape(layout.input.string()) << "\",\n";
    out << "    \"prof_dir\": \"" << json_escape(layout.prof_dir.string()) << "\",\n";
    out << "    \"device_dir\": \"" << json_escape(layout.device_dir.filename().string()) << "\",\n";
    out << "    \"hwts_frequency_mhz\": " << frequency_mhz << ",\n";
    out << "    \"rank\": " << (rank ? std::to_string(*rank) : "null") << ",\n";
    out << "    \"segment\": " << (segment ? std::to_string(*segment) : "null") << ",\n";
    out << "    \"declared_first_step\": " << (first_step ? std::to_string(*first_step) : "null") << ",\n";
    out << "    \"declared_last_step\": " << (last_step ? std::to_string(*last_step) : "null") << "\n";
    out << "  },\n";
    out << "  \"coverage\": {\n";
    out << "    \"ts_track_files\": " << layout.ts_track_files.size() << ",\n";
    out << "    \"ts_track_available\": "
        << (layout.ts_track_files.empty() ? "false" : "true") << ",\n";
    out << "    \"step_trace_available\": "
        << (step_trace_available ? "true" : "false") << ",\n";
    out << "    \"step_trace_complete\": "
        << (step_trace_complete ? "true" : "false") << ",\n";
    out << "    \"expected_step_pairs\": "
        << (expected_step_pairs ? std::to_string(*expected_step_pairs) : "null") << ",\n";
    out << "    \"ts_track_bytes\": " << step_result.input.bytes << ",\n";
    out << "    \"ts_track_records\": " << step_result.input.records << ",\n";
    out << "    \"step_trace_records\": " << step_result.step_trace_records << ",\n";
    out << "    \"step_pairs\": " << step_result.steps.size() << ",\n";
    out << "    \"step_unmatched_starts\": " << step_result.unmatched_starts << ",\n";
    out << "    \"step_unmatched_ends\": " << step_result.unmatched_ends << ",\n";
    out << "    \"stars_files\": " << layout.stars_files.size() << ",\n";
    out << "    \"stars_bytes\": " << stars_result.input.bytes << ",\n";
    out << "    \"stars_records\": " << stars_result.input.records << ",\n";
    out << "    \"acsq_starts\": " << stars_result.acsq_starts << ",\n";
    out << "    \"acsq_ends\": " << stars_result.acsq_ends << ",\n";
    out << "    \"matched_device_tasks\": " << stars_result.tasks.size() << ",\n";
    out << "    \"task_unmatched_starts\": " << stars_result.unmatched_starts << ",\n";
    out << "    \"task_unmatched_ends\": " << stars_result.unmatched_ends << ",\n";
    out << "    \"step_assigned_tasks\": " << assigned_tasks << ",\n";
    const double assignment_ratio = stars_result.tasks.empty() ? 0.0 :
        static_cast<double>(assigned_tasks) / static_cast<double>(stars_result.tasks.size());
    const double name_assignment_ratio = stars_result.tasks.empty() ? 0.0 :
        static_cast<double>(named_device_tasks) / static_cast<double>(stars_result.tasks.size());
    const double compute_name_coverage = eligible_compute_tasks == 0 ? 0.0 :
        static_cast<double>(named_compute_tasks) / static_cast<double>(eligible_compute_tasks);
    out << "    \"step_assignment_ratio\": " << assignment_ratio << ",\n";
    out << "    \"device_task_flips\": " << step_result.task_flips.size() << ",\n";
    out << "    \"host_metadata_available\": " << (host_result.available ? "true" : "false") << ",\n";
    out << "    \"api_event_files\": " << layout.api_event_files.size() << ",\n";
    out << "    \"api_event_bytes\": " << host_result.api_input.bytes << ",\n";
    out << "    \"api_event_records\": " << host_result.api_input.records << ",\n";
    out << "    \"unsupported_api_event_points\": " << host_result.api_event_records << ",\n";
    out << "    \"node_intervals\": " << host_result.node_intervals << ",\n";
    out << "    \"hccl_intervals\": " << host_result.hccl_intervals << ",\n";
    out << "    \"hccl_op_info_files\": " << layout.hccl_op_info_files.size() << ",\n";
    out << "    \"hccl_op_info_bytes\": " << host_result.hccl_op_input.bytes << ",\n";
    out << "    \"hccl_op_records\": " << host_result.hccl_ops.size() << ",\n";
    out << "    \"hccl_ops_without_node\": " << host_result.hccl_ops_without_node << ",\n";
    out << "    \"task_track_files\": " << layout.task_track_files.size() << ",\n";
    out << "    \"task_track_bytes\": " << host_result.task_input.bytes << ",\n";
    out << "    \"task_track_records\": " << host_result.task_input.records << ",\n";
    out << "    \"host_task_flips\": " << host_result.flip_task_tracks << ",\n";
    out << "    \"kernel_task_tracks\": " << host_result.kernel_task_tracks << ",\n";
    out << "    \"host_kernel_records\": " << host_result.kernels.size() << ",\n";
    out << "    \"host_kernel_records_without_name\": " << host_result.kernel_tracks_without_name << ",\n";
    out << "    \"duplicate_host_kernel_keys\": " << duplicate_host_kernel_keys << ",\n";
    out << "    \"named_device_tasks\": " << named_device_tasks << ",\n";
    out << "    \"unmatched_host_kernels\": " << unmatched_host_kernels << ",\n";
    out << "    \"name_assignment_ratio\": " << name_assignment_ratio << ",\n";
    out << "    \"eligible_compute_tasks\": " << eligible_compute_tasks << ",\n";
    out << "    \"named_compute_tasks\": " << named_compute_tasks << ",\n";
    out << "    \"compute_name_coverage\": " << compute_name_coverage << "\n";
    out << "  },\n";
    out << "  \"steps\": [\n";
    for (std::size_t i = 0; i < step_result.steps.size(); ++i) {
        const auto& step = step_result.steps[i];
        std::optional<std::int64_t> global_step;
        const auto step_offset = static_cast<std::int64_t>(i);
        if (step_trace_complete && first_step
                && (!last_step || *first_step + step_offset <= *last_step)) {
            global_step = *first_step + step_offset;
        }
        out << "    {\"global_step\": "
            << (global_step ? std::to_string(*global_step) : "null")
            << ", \"index_id\": " << step.index_id
            << ", \"model_id\": " << step.model_id
            << ", \"start_syscnt\": " << step.start
            << ", \"end_syscnt\": " << step.end
            << ", \"duration_ticks\": " << (step.end - step.start)
            << ", \"duration_ns\": " << ticks_to_ns(step.end - step.start, frequency_mhz) << "}";
        out << (i + 1 == step_result.steps.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"device_task_types\": [\n";
    std::size_t type_index = 0;
    for (const auto& item : per_type) {
        const auto& stats = item.second;
        const auto& durations = stats.durations;
        out << "    {\"task_type_id\": " << static_cast<unsigned int>(stats.type)
            << ", \"task_type\": \"" << task_type_name(stats.type) << "\""
            << ", \"count\": " << durations.size()
            << ", \"total_ns\": " << ticks_to_ns(stats.total_ticks, frequency_mhz)
            << ", \"min_ns\": " << ticks_to_ns(durations.front(), frequency_mhz)
            << ", \"p50_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.50), frequency_mhz)
            << ", \"p95_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.95), frequency_mhz)
            << ", \"p99_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.99), frequency_mhz)
            << ", \"max_ns\": " << ticks_to_ns(durations.back(), frequency_mhz) << "}";
        out << (++type_index == per_type.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"kernels\": [\n";
    std::size_t kernel_index = 0;
    for (const auto& item : named_kernel_stats) {
        const auto& stats = item.second;
        const auto& durations = stats.durations;
        out << "    {\"name\": \"" << json_escape(stats.name) << "\""
            << ", \"name_source\": \"" << json_escape(stats.name_source) << "\""
            << ", \"host_task_type\": \"" << json_escape(stats.host_task_type) << "\""
            << ", \"stream_id\": " << stats.stream_id
            << ", \"device_task_type_id\": " << static_cast<unsigned int>(stats.device_task_type)
            << ", \"device_task_type\": \"" << task_type_name(stats.device_task_type) << "\""
            << ", \"count\": " << durations.size()
            << ", \"total_ns\": " << ticks_to_ns(stats.total_ticks, frequency_mhz)
            << ", \"min_ns\": " << ticks_to_ns(durations.front(), frequency_mhz)
            << ", \"p50_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.50), frequency_mhz)
            << ", \"p95_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.95), frequency_mhz)
            << ", \"p99_ns\": " << ticks_to_ns(quantile_sorted(durations, 0.99), frequency_mhz)
            << ", \"max_ns\": " << ticks_to_ns(durations.back(), frequency_mhz) << "}";
        out << (++kernel_index == named_kernel_stats.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"hccl_ops\": [\n";
    std::size_t hccl_index = 0;
    for (const auto& item : hccl_stats) {
        const auto& stats = item.second;
        const auto& durations = stats.host_durations;
        out << "    {\"op_name\": \"" << json_escape(stats.op_name) << "\""
            << ", \"data_type\": \"" << json_escape(stats.data_type) << "\""
            << ", \"algorithm\": \"" << json_escape(stats.algorithm) << "\""
            << ", \"group_hash\": " << stats.group_hash
            << ", \"group_name\": \"" << json_escape(stats.group_name) << "\""
            << ", \"element_count\": " << stats.element_count
            << ", \"relay\": " << (stats.relay ? "true" : "false")
            << ", \"retry\": " << (stats.retry ? "true" : "false")
            << ", \"count\": " << durations.size();
        if (durations.empty()) {
            out << ", \"host_launch_min_ns\": null, \"host_launch_total_ns\": null"
                << ", \"host_launch_p50_ns\": null"
                << ", \"host_launch_p95_ns\": null, \"host_launch_p99_ns\": null"
                << ", \"host_launch_max_ns\": null}";
        } else {
            std::uint64_t total_host_duration = 0;
            for (const auto duration : durations) {
                total_host_duration += duration;
            }
            out << ", \"host_launch_min_ns\": " << durations.front()
                << ", \"host_launch_total_ns\": " << total_host_duration
                << ", \"host_launch_p50_ns\": " << quantile_sorted(durations, 0.50)
                << ", \"host_launch_p95_ns\": " << quantile_sorted(durations, 0.95)
                << ", \"host_launch_p99_ns\": " << quantile_sorted(durations, 0.99)
                << ", \"host_launch_max_ns\": " << durations.back() << "}";
        }
        out << (++hccl_index == hccl_stats.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"top_slowest_tasks\": [\n";
    for (std::size_t i = 0; i < slowest.size(); ++i) {
        const auto& task = *slowest[i];
        const auto host_kernel = host_kernel_by_task.find({task.stream_id, task.task_id, task.batch_id});
        out << "    {\"stream_id\": " << task.stream_id
            << ", \"task_id\": " << task.task_id
            << ", \"batch_id\": " << task.batch_id
            << ", \"acc_id\": " << static_cast<unsigned int>(task.acc_id)
            << ", \"task_type_id\": " << static_cast<unsigned int>(task.task_type)
            << ", \"task_type\": \"" << task_type_name(task.task_type) << "\""
            << ", \"step_index\": " << task.step_index
            << ", \"start_syscnt\": " << task.start
            << ", \"duration_ticks\": " << task.duration_ticks()
            << ", \"duration_ns\": " << ticks_to_ns(task.duration_ticks(), frequency_mhz)
            << ", \"kernel_name\": ";
        if (host_kernel == host_kernel_by_task.end()) {
            out << "null";
        } else {
            out << "\"" << json_escape(host_kernel->second->name) << "\"";
        }
        out << "}";
        out << (i + 1 == slowest.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

struct Options {
    fs::path input;
    std::optional<fs::path> output;
    std::size_t top_k = 20;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: fast_parse [--output PATH] [--top-k N] RANK_SEGMENT_DIR\n";
            std::exit(0);
        }
        if (arg == "--output") {
            if (++i >= argc) {
                throw ParseError("--output requires a path");
            }
            options.output = fs::path(argv[i]);
            continue;
        }
        if (arg == "--top-k") {
            if (++i >= argc) {
                throw ParseError("--top-k requires a non-negative integer");
            }
            try {
                options.top_k = static_cast<std::size_t>(std::stoull(argv[i]));
            } catch (const std::exception&) {
                throw ParseError("invalid --top-k value: " + std::string(argv[i]));
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw ParseError("unknown option: " + arg);
        }
        if (!options.input.empty()) {
            throw ParseError("only one input directory is allowed");
        }
        options.input = fs::path(arg);
    }
    if (options.input.empty()) {
        throw ParseError("missing input directory");
    }
    return options;
}

void write_atomic(const fs::path& output, const std::string& contents) {
    const fs::path absolute = fs::absolute(output).lexically_normal();
    const fs::path parent = absolute.parent_path();
    if (!parent.empty() && !fs::is_directory(parent)) {
        throw ParseError("output parent is not a directory: " + parent.string());
    }
    const fs::path temporary = absolute.string() + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw ParseError("cannot create " + temporary.string());
        }
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        out.flush();
        if (!out) {
            throw ParseError("cannot write " + temporary.string());
        }
    }
    std::error_code error;
    fs::rename(temporary, absolute, error);
    if (error) {
        fs::remove(temporary);
        throw ParseError("cannot rename output to " + absolute.string() + ": " + error.message());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto begin = std::chrono::steady_clock::now();
        const InputLayout layout = discover_layout(options.input);
        const double frequency_mhz = read_hwts_frequency_mhz(layout.info_json);
        const StepParseResult steps = parse_steps(layout.ts_track_files);
        StarsParseResult stars = parse_stars(layout.stars_files);
        assign_device_batches(stars.tasks, steps.task_flips);
        const HostParseResult host = parse_host_names(layout);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - begin).count();
        const std::string output = render_json(
            layout, frequency_mhz, steps, stars, host, elapsed, options.top_k);
        if (options.output) {
            write_atomic(*options.output, output);
        } else {
            std::cout << output;
        }
        return 0;
    } catch (const ParseError& error) {
        std::cerr << "fast_parse: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "fast_parse: unexpected error: " << error.what() << '\n';
        return 3;
    }
}
