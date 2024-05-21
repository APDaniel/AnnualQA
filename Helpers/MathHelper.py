import math


class MathHelper:
    def __init__(self,logger):
        self.logger=logger
        
    def rotateCoordinate(self,x,y,angleDeg,center_x,center_y):
        angleRad=math.radians(angleDeg)
        x-=center_x
        y-=center_y
        x_rot=x*math.cos(angleRad)-y*math.sin(angleRad)+center_x
        y_rot=x*math.sin(angleRad)+y*math.cos(angleRad)+center_y
        return x_rot,y_rot
