module DomainTypes

export TradePair, LegStage, leg_stage_symbol, leg_stage_label,
       ChunkId, ChunkPaths, default_julia_log_path

"""
    TradePair

Alias for the `(from_asset, to_asset)` tuple used throughout the simulator.
"""
const TradePair = NTuple{2,Int}

"""
    @enum LegStage DIR1 DIR2

Identifies which half of a candle we are processing. Mirrors the C++ DIR1/DIR2
labels used in logs.
"""
@enum LegStage DIR1 DIR2

leg_stage_symbol(stage::LegStage) = stage === DIR1 ? :dir1 : :dir2
leg_stage_label(stage::LegStage) = stage === DIR1 ? "DIR1" : "DIR2"

"""
    struct ChunkId

Wraps a chunk directory name (`chunkXXXXX`) and validates the prefix.
"""
struct ChunkId
    value::String
    function ChunkId(val::AbstractString)
        str = String(val)
        startswith(str, "chunk") || throw(ArgumentError("chunk id must start with 'chunk', got $(str)"))
        return new(str)
    end
end

Base.show(io::IO, id::ChunkId) = print(io, id.value)
Base.String(id::ChunkId) = id.value

"""
    struct ChunkPaths

Holds canonical paths for a chunk directory (metadata, config, results, logs).
"""
struct ChunkPaths
    id::ChunkId
    dir::String
    metadata::String
    chunk_config::String
    results::String
    cpp_log::String
end

function ChunkPaths(root::AbstractString, chunk::AbstractString)
    dir = isdir(chunk) ? normpath(chunk) : normpath(joinpath(root, chunk))
    isdir(dir) || error("Chunk directory $(dir) not found")
    id = ChunkId(basename(dir))
    metadata = joinpath(dir, "metadata.json")
    isfile(metadata) || error("metadata.json missing in $(dir)")
    chunk_cfg = joinpath(dir, "chunk-config.json")
    results = joinpath(dir, "results.json")
    cpp_log = joinpath(dir, "cpp_log.jsonl")
    return ChunkPaths(id, dir, metadata, chunk_cfg, results, cpp_log)
end

default_julia_log_path(paths::ChunkPaths) = joinpath(paths.dir, "julia_log.$(String(paths.id)).jsonl")

end # module
