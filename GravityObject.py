import newt.qlm as qlm
import newt.translations as translations

class GravityObject:
	def __init__(self, name, components, orientation, position):
		self.name = name
		self.components = components  #this is a list. Not sure how to deal with typing.
		self.orientation = orientation  
		self.position = position

	#Stub, doesn't rotate moments
	@staticmethod
	def rotateMoments(moments, orientation):
		return moments
	
	@staticmethod
	def translateMoments(moments, position):
		return translations.translate_qlm(moments, position)

	def computeMoments(self, l = 9):
		moments = 0
		for i in self.components:
			moments = moments + \
				self.translateMoments( 
					self.rotateMoments(
						i.computeMoments( l ) , 
					i.orientation) ,
				i.position )

		return moments



class Annulus(GravityObject):
	def __init__ (self, name, mass, height, inner_radius, outer_radius, orientation, position, mean_angle, half_angle, xgrid = 5, zgrid = 5):
		self.name = name
		self.mass = mass
		self.height = height
		self.inner_radius = inner_radius
		self.outer_radius = outer_radius
		self.orientation = orientation
		self.position = position
		self.mean_angle = mean_angle
		self.half_angle = half_angle
		self.xgrid = xgrid
		self.zgrid = zgrid

	def computeMoments(self, l = 0):
		return qlm.annulus( l, self.mass, self.height, self.inner_radius,
					self.outer_radius, self.mean_angle,
					self.half_angle)



import numpy as np

b = Annulus("bob", 6 , 2, 1, 2, np.array([3,2,1]) , np.array([3, 2, 1]), 0,np.pi )


print ("b")
print(b.computeMoments(2))

groupedComponent = GravityObject("group", [b,b], [3,2,1], [3,2,1])


print("groupedComponent")
print(groupedComponent.computeMoments(2))

