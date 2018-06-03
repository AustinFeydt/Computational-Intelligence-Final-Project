#Represents a vector containing the vector's petal width, length,
#and name of the iris
class IrisVector:
	def __init__(self, joined_row):
		components = joined_row.split(",")
		self.pl = float(components[2])
		self.pw = float(components[3])
		self.name = components[4]