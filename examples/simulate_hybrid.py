from hopp.tools.hopp_interface import HoppInterface


hi = HoppInterface("inputs/wind_solar.yaml")

hi.simulate(project_life=25)
hi.parse_output()
hi.print_output()

print(hi.hopp.system.annual_energies)
print(hi.hopp.system.net_present_values)
