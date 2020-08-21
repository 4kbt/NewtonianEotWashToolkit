class GravityObject:
	def __init__(self, name, components, orientation, position):
		self.name = name
		self.components = components  #this is a list. Not sure how to deal with typing.
		self.orientation = orientation  
		self.position = position

	def computeMoments(self, l = 0):
		moments = 0
		for i in self.components:
			moments = moments + i.computeMoments()

		return moments


class Annulus(GravityObject):
	def __init__ (self, name, mass, inner_radius, outer_radius, orientation, position, xgrid = 5, zgrid = 5):
		self.name = name
		self.mass = mass
		self.inner_radius = inner_radius
		self.outer_radius = outer_radius
		self.orientation = orientation
		self.position = position
		self.xgrid = xgrid
		self.zgrid = zgrid

	def computeMoments(self, l = 0):
		return self.mass



import numpy as np

b = Annulus("bob", 6, 1, 2, np.array([3,2,1]) , np.array([3, 2, 1]), 4 ,4 )

print(b.computeMoments())

groupedComponent = GravityObject("group", [b,b], [3,2,1], [3,2,1])

print(groupedComponent.computeMoments())

