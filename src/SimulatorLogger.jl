module SimulatorLogger

using ..SimulatorCore: Trader
import ..Instrumentation

const Instr = Instrumentation

export LoggerAdapter, build_logger, logger_debug, logger_sink,
       log_preleg!, log_preleg_limit!, log_step!, log_leg!, log_tweak!,
       maybe_step_probe, emit_step_probe!, trace_step_iter!, trace_step_result!,
       trace_dir2!, compute_debug_options

struct LoggerAdapter
    debug::Instr.DebugOptions
end

logger_debug(adapter::LoggerAdapter) = adapter.debug
logger_sink(adapter::LoggerAdapter) = adapter.debug.logger

function compute_debug_options(logger::Union{Nothing,Instr.TradeLogger},
                               debug::Union{Nothing,Instr.DebugOptions})
    if debug === nothing
        return Instr.default_debug_options(; logger=logger)
    end
    if logger === nothing
        return debug
    end
    return Instr.with_logger(debug, logger)
end

build_logger(logger::Union{Nothing,Instr.TradeLogger},
             debug::Union{Nothing,Instr.DebugOptions}) = LoggerAdapter(
    compute_debug_options(logger, debug)
)

log_preleg!(adapter::LoggerAdapter, args...) =
    Instr.log_preleg_event(logger_sink(adapter), args...)

log_preleg_limit!(adapter::LoggerAdapter, args...) =
    Instr.log_preleg_limit_event(logger_sink(adapter), args...)

log_step!(adapter::LoggerAdapter, args...) =
    Instr.log_step_event(logger_sink(adapter), args...)

log_leg!(adapter::LoggerAdapter, args...) =
    Instr.log_leg_event(logger_sink(adapter), args...)

log_tweak!(adapter::LoggerAdapter, args...) =
    Instr.log_tweak_event(logger_sink(adapter), args...)

maybe_step_probe(adapter::LoggerAdapter, args...) =
    Instr.maybe_step_probe_context(logger_debug(adapter), args...)

trace_dir2!(adapter::LoggerAdapter, ctx::NamedTuple) =
    Instr.trace_dir2_execution(logger_debug(adapter), ctx)

trace_step_iter!(adapter::LoggerAdapter, stage, phase, ctx) =
    Instr.trace_step_iter(logger_debug(adapter), stage, phase, ctx)

trace_step_result!(adapter::LoggerAdapter, stage, ctx) =
    Instr.trace_step_result(logger_debug(adapter), stage, ctx)

emit_step_probe!(adapter::LoggerAdapter, ctx, trader::Trader, args...) =
    Instr.emit_step_probe(logger_debug(adapter), ctx, trader, args...)

end # module
