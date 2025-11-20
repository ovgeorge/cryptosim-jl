module ChunkLoader

using JSON3

using ..DataIO
using ..DomainTypes: ChunkPaths

export ChunkData, load_chunk, read_expected, ensure_data_sources!

struct ChunkData
    config::DataIO.SimulationConfig
    price_vector::Vector{Float64}
    trades::Vector{DataIO.CPPTrade}
    metadata::JSON3.Object
end

function ensure_data_sources!(cfg::DataIO.ConfigFile, data_dir::AbstractString)
    for source in cfg.datafiles
        file = endswith(lowercase(source), ".json") ? source : string(source, ".json")
        path = joinpath(data_dir, file)
        isfile(path) || error("Missing data source $(path). Place the raw candle JSON under $(data_dir).")
    end
end

function read_metadata(paths::ChunkPaths)
    open(paths.metadata, "r") do io
        return JSON3.read(io)
    end
end

function price_vector_from_trades(n::Int, trades)
    prices = ones(Float64, n)
    seen = falses(n)
    seen[1] = true
    for trade in trades
        a, b = trade.pair
        if a == 0 && b + 1 <= n
            prices[b + 1] = trade.close
            seen[b + 1] = true
        elseif b == 0 && a + 1 <= n && trade.close != 0
            prices[a + 1] = 1 / trade.close
            seen[a + 1] = true
        end
        all(seen) && break
    end
    return prices
end

parse_trim(flag::AbstractString) = parse(Int, replace(flag, "trim" => ""))

function load_chunk(paths::ChunkPaths; data_dir::AbstractString=DataIO.DEFAULT_DATA_DIR)
    cfg_file = if isfile(paths.chunk_config)
        DataIO.load_config(paths.chunk_config)
    else
        metadata = read_metadata(paths)
        cfg_name = haskey(metadata, "config") ? String(metadata["config"]) : "single-run.json"
        DataIO.load_config(joinpath(paths.dir, cfg_name))
    end
    ensure_data_sources!(cfg_file, data_dir)
    metadata = read_metadata(paths)
    trades = if haskey(metadata, "trim_flag")
        trim = parse_trim(String(metadata["trim_flag"]))
        DataIO.build_cpp_trades(cfg_file; data_dir=data_dir, trim=trim)
    else
        DataIO.build_cpp_trades(cfg_file; data_dir=data_dir)
    end
    cfg = cfg_file.configurations[1]
    price_vec = price_vector_from_trades(cfg.n, trades)
    return ChunkData(cfg, price_vec, trades, metadata)
end

function read_expected(paths::ChunkPaths)
    results = open(joinpath(paths.dir, "results.json"), "r") do io
        JSON3.read(io)
    end
    metrics = results["configuration"][1]["Result"]
    return (
        volume = Float64(metrics["volume"]),
        slippage = Float64(metrics["slippage"]),
        liquidity_density = Float64(metrics["liq_density"]),
        apy = Float64(metrics["APY"]),
    )
end

end # module
