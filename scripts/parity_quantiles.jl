#!/usr/bin/env julia

"""
Read `reports/ethusdt_full_chunk_summary.jsonl` and print relative-error
quantiles for the key metrics. Intended to run after
`scripts/full_parity_report.sh`.
"""

using JSON3

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "CryptoSim.jl"))
using .CryptoSim

const Summary = CryptoSim.ChunkSummary
const DEFAULT_REPORT_PATH = normpath(joinpath(PROJECT_ROOT, "reports", "ethusdt_full_chunk_summary.jsonl"))

mutable struct Options
    input::String
    markdown::Union{Nothing,String}
    title::String
    dataset::Union{Nothing,String}
    dataset_sha::Union{Nothing,String}
    dataset_path::Union{Nothing,String}
    chunk_root::Union{Nothing,String}
    runner::Union{Nothing,String}
    notes::Vector{String}
end

function Options()
    return Options(
        DEFAULT_REPORT_PATH,
        nothing,
        "Parity Report",
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        String[],
    )
end

function usage()
    println("""
Usage: parity_quantiles.jl [--input PATH] [--markdown PATH] [options...]

Options:
  --input PATH          Summary JSONL path (default: $(DEFAULT_REPORT_PATH))
  --markdown PATH       Write Markdown report to PATH
  --title TEXT          Markdown document title
  --dataset NAME        Dataset name
  --dataset-path PATH   Dataset path (used in markdown bullet)
  --dataset-sha SHA     Dataset sha256
  --chunk-root DIR      Chunk root directory for provenance
  --runner TEXT         Runner description to record in markdown
  --note TEXT           Additional note (may be repeated)
  -h, --help            Show this message

The script always prints the quantile table to stdout. When --markdown is
provided it will also write a Markdown report using the supplied metadata.
""")
end

function parse_args()
    opts = Options()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif startswith(arg, "--input=")
            opts.input = abspath(split(arg, '='; limit=2)[2])
        elseif arg == "--input"
            i += 1; i > length(ARGS) && error("--input requires a path")
            opts.input = abspath(ARGS[i])
        elseif startswith(arg, "--markdown=")
            opts.markdown = abspath(split(arg, '='; limit=2)[2])
        elseif arg == "--markdown"
            i += 1; i > length(ARGS) && error("--markdown requires a path")
            opts.markdown = abspath(ARGS[i])
        elseif startswith(arg, "--title=")
            opts.title = split(arg, '='; limit=2)[2]
        elseif arg == "--title"
            i += 1; i > length(ARGS) && error("--title requires text")
            opts.title = ARGS[i]
        elseif startswith(arg, "--dataset=")
            opts.dataset = split(arg, '='; limit=2)[2]
        elseif arg == "--dataset"
            i += 1; i > length(ARGS) && error("--dataset requires text")
            opts.dataset = ARGS[i]
        elseif startswith(arg, "--dataset-path=")
            opts.dataset_path = split(arg, '='; limit=2)[2]
        elseif arg == "--dataset-path"
            i += 1; i > length(ARGS) && error("--dataset-path requires text")
            opts.dataset_path = ARGS[i]
        elseif startswith(arg, "--dataset-sha=")
            opts.dataset_sha = split(arg, '='; limit=2)[2]
        elseif arg == "--dataset-sha"
            i += 1; i > length(ARGS) && error("--dataset-sha requires text")
            opts.dataset_sha = ARGS[i]
        elseif startswith(arg, "--chunk-root=")
            opts.chunk_root = abspath(split(arg, '='; limit=2)[2])
        elseif arg == "--chunk-root"
            i += 1; i > length(ARGS) && error("--chunk-root requires a path")
            opts.chunk_root = abspath(ARGS[i])
        elseif startswith(arg, "--runner=")
            opts.runner = split(arg, '='; limit=2)[2]
        elseif arg == "--runner"
            i += 1; i > length(ARGS) && error("--runner requires text")
            opts.runner = ARGS[i]
        elseif startswith(arg, "--note=")
            push!(opts.notes, split(arg, '='; limit=2)[2])
        elseif arg == "--note"
            i += 1; i > length(ARGS) && error("--note requires text")
            push!(opts.notes, ARGS[i])
        else
            opts.input = abspath(arg)
        end
        i += 1
    end
    return opts
end

function load_summaries(report_path::AbstractString)
    summaries = Summary.ChunkSummary[]
    open(report_path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            obj = JSON3.read(line, Dict{String,Any})
            push!(summaries, Summary.from_json_dict(obj))
        end
    end
    return summaries
end

function build_metadata(opts::Options)
    pairs = Pair{Symbol,Any}[]
    opts.dataset !== nothing && push!(pairs, :dataset_name => opts.dataset)
    opts.dataset_path !== nothing && push!(pairs, :dataset_path => opts.dataset_path)
    opts.dataset_sha !== nothing && push!(pairs, :dataset_sha => opts.dataset_sha)
    opts.chunk_root !== nothing && push!(pairs, :chunk_root => opts.chunk_root)
    opts.runner !== nothing && push!(pairs, :runner => opts.runner)
    !isempty(opts.notes) && push!(pairs, :notes => copy(opts.notes))
    return isempty(pairs) ? NamedTuple() : NamedTuple(pairs)
end

function main()
    opts = parse_args()
    isfile(opts.input) || error("Summary file $(opts.input) not found. Run scripts/full_parity_report.sh first.")
    summaries = load_summaries(opts.input)
    isempty(summaries) && error("No rows found in $(opts.input)")
    stats = Summary.build_stats(summaries)
    println("Chunk logs matching C++: $(stats.log_matches)/$(stats.total)")
    print(Summary.format_quantile_table(stats))
    if opts.markdown !== nothing
        metadata = build_metadata(opts)
        md = Summary.render_markdown_report(stats; title=opts.title, metadata=metadata)
        open(opts.markdown, "w") do io
            write(io, md)
        end
        println("\n[parity-quantiles] wrote Markdown to $(opts.markdown)")
    end
end

main()
