module CryptoSim

# Top-level exports for the Julia rewrite.
include("DataIO.jl")
include("Preprocessing.jl")
include("Metrics.jl")
include("Tweaks.jl")
include("Simulator.jl")
include("CLI.jl")

using .DataIO
using .Preprocessing
using .Tweaks
using .Metrics
using .Simulator
using .CLI

export DataIO, Preprocessing, Simulator, Tweaks, Metrics, CLI, run_cli

"""
    run_cli(args=Base.ARGS)

Parses CLI flags, loads the input JSON, and returns both the CLI options and config bundle.
"""
function run_cli(args::Vector{String}=Vector{String}(Base.ARGS))
    opts = CLI.parse_cli_args(args)
    config = DataIO.load_config(opts.input_path)
    return (; options = opts, config)
end

end # module
