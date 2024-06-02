from pydbus import SessionBus
from gi.repository import GLib
from pydbus.generic import signal
from pydbus_module import VehicleControl

class VehicleControlDBusService:
    """
    D-Bus Service Interface for VehicleControl.
    """
    dbus = """
    <node>
        <interface name='com.team2.VehicleControl'>
            <method name='SetSteering'>
                <arg type='d' name='value' direction='in'/>
            </method>
            <method name='SetThrottle'>
                <arg type='d' name='value' direction='in'/>
            </method>
            <property name='Steering' type='d' access='read'/>
            <property name='Throttle' type='d' access='read'/>
            <signal name='SteeringChanged'>
                <arg type='d' name='newSteering'/>
            </signal>
            <signal name='ThrottleChanged'>
                <arg type='d' name='newThrottle'/>
            </signal>
        </interface>
    </node>
    """

    def __init__(self):
        self.vehicle = VehicleControl()

    def SetSteering(self, value):
        self.vehicle.steering = value
        self.SteeringChanged(value)
        return f"Steering set to {value}"

    def SetThrottle(self, value):
        self.vehicle.throttle = value
        self.ThrottleChanged(value)
        return f"Throttle set to {value}"
    
    SteeringChanged = signal()
    ThrottleChanged = signal()

    @property
    def Steering(self):
        return self.vehicle.steering

    @property
    def Throttle(self):
        return self.vehicle.throttle
    
# if __name__ == "__main__":
#     bus = SessionBus()
#     service = VehicleControlDBusService()
#     bus.publish("com.example.VehicleControl", service)
#     print("VehicleControl D-Bus service running.")
#     loop = GLib.MainLoop()
#     loop.run()
