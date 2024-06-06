class VehicleControl:
    def __init__(self):
        self._steering = 0.0
        self._throttle = 0.0
    
    @property
    def steering(self):
        return self._steering
    
    @steering.setter
    def steering(self, value):
        self._steering = value
        #For debugging
        print(f"Steering set to {self._steering}")
        
    @property
    def throttle(self):
        return self._throttle
    
    @throttle.setter
    def throttle(self, value):
        self._throttle = value
        #For debugging
        print(f"Speed set to {self._throttle}")    
    