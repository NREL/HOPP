import PySAM.BatteryStateful as BatteryModel
import PySAM.BatteryTools as BatteryTools
system_model = BatteryModel.default("NMCGraphite")
BatteryTools.battery_model_sizing(system_model,
                                  0.,
                                  10,
                                  500)
system_model.value("control_mode", 0.0)
system_model.value("dt_hr", 1)
system_model.value("input_current", 4.0)
system_model.setup()

