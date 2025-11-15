struct LoggerAdapter
    debug::Instr.DebugOptions
end

logger_debug(adapter::LoggerAdapter) = adapter.debug
logger_sink(adapter::LoggerAdapter) = adapter.debug.logger

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
