module CLI

export CLIOptions, parse_cli_args, DEFAULT_TRIM

const DEFAULT_TRIM = 1_000_000

"""
    struct CLIOptions

Represents the subset of arguments understood by the legacy `simu` binary.
"""
struct CLIOptions
    trim::Union{Nothing,Int}
    threads::Int
    input_path::String
    output_path::String
end

"""
    parse_cli_args(args::Vector{String}) -> CLIOptions

Parses the `trim`, `threads=#`, `[in] [out]` pattern used by `cryptopool-simulator/simu`.
"""
function parse_cli_args(args::Vector{String})
    idx = 1
    trim::Union{Nothing,Int} = nothing
    threads = 1
    input_path = "sample_in.json"
    output_path = "sample_out.json"
    while idx <= length(args)
        arg = args[idx]
        if startswith(arg, "trim")
            trim = if length(arg) == 4
                DEFAULT_TRIM
            else
                parse(Int, arg[5:end])
            end
            idx += 1
        elseif startswith(arg, "threads=")
            threads = max(parse(Int, arg[9:end]), 1)
            idx += 1
        else
            break
        end
    end
    if idx <= length(args)
        input_path = args[idx]
        idx += 1
    end
    if idx <= length(args)
        output_path = args[idx]
    end
    return CLIOptions(trim, threads, input_path, output_path)
end

end # module
