#what is Numpy?
'''NumPy is a Python library used for working with arrays.
It also has functions for working in domain of linear algebra, fourier transform, and matrices.
NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.
NumPy stands for Numerical Python.'''
#why use numpy?
'''In Python we have lists that serve the purpose of arrays, but they are slow to process.
NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.
The array object in NumPy is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy.
Arrays are very frequently used in data science, where speed and resources are very important.'''
#Why is NumPy Faster Than Lists?
'''NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
This behavior is called locality of reference in computer science.
This is the main reason why NumPy is faster than lists. Also it is optimized to work with latest CPU architectures.'''
#Example
import numpy as np
 # Creating array object
arr = np.array( [[ 1, 2, 3],
                 [ 4, 2, 5]] )
# Printing type of arr object
print("Array is of type: ", type(arr))
# Printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)
# Printing shape of array
print("Shape of array: ", arr.shape)
# Printing size (total number of elements) of array
print("Size of array: ", arr.size)
# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)
'''Output:
Array is of type:  
No. of dimensions:  2
Shape of array:  (2, 3)
Size of array:  6
Array stores elements of type:  int64 '''

# How to traverse file system in python?
#suppose we have file structure like tree and we want to taverse all it's branches completely from top to bottom
'''we use os.walk() method and How does os.walk() work in python ?
OS.walk() generate the file names in a directory tree by walking the tree either top-down or bottom-up.
For each directory in the tree rooted at directory top (including top itself), 
it yields a 3-tuple (dirpath, dirnames, filenames).
root : Prints out directories only from what you specified.
dirs : Prints out sub-directories from root.
files : Prints out all files from root and directories.'''
#Example
'''
import os
for (root,dirs,files) in os.walk('Test', topdown=True):
        print (root)
        print (dirs)
        print (files)
        '''
        
#os.path.basename() method
'''os.path module is sub module of OS module in Python used for common path name manipulation.
os.path.basename() method in Python is used to get the base name in specified path. 
This method internally use os.path.split() method to split the specified path into a pair (head, tail). 
os.path.basename() method returns the tail part after splitting the specified path into (head, tail) pair. '''
#Example
import os.path
path = '/home/User/Documents'
path2 = 'file.txt'
# Above specified path will be splitted into (head, tail) pair as ('/home/User', 'Documents') and ('','file.text')
# Get the base name of the specified path
basename = os.path.basename(path)
basename = os.path.basename(path2)
print(basename)    #-> Documents    
print(basename)    #-> file.text  

#os.path.join() method
'''os.path.join() method in Python join one or more path components intelligently. 
This method concatenates various path components with exactly one directory separator (‘/’) 
following each non-empty part except the last path component. 
If the last path component to be joined is empty then a directory separator (‘/’) is put at the end. 
If a path component represents an absolute path, then all previous components joined are discarded and 
joining continues from the absolute path component.'''  
#Example
import os
path = "/home"
# Join various path components
print(os.path.join(path, "User/Desktop", "file.txt"))   
'''/home/User/Desktop/file.txt'''
path = "User/Documents"
print(os.path.join(path, "/home", "file.txt"))          
'''/home/file.txt'''
# In above example '/home' represents an absolute path so all previous components i.e 
# User / Documents are thrown away and joining continues from the absolute path component i.e / home.
path = "/User"
print(os.path.join(path, "Downloads", "file.txt", "/home"))  
'''/home'''
# In above example '/User' and '/home' both represents an absolute path but '/home' is the last value
# so all previous components before '/home' will be discarded and joining will continue from '/home'
path = "/home"
print(os.path.join(path, "User/Public/", "Documents", ""))   
'''/home/User/Public/Documents/'''
# In above example the last path component is empty so a directory separator ('/') will be put at the end along with the concatenated value

#LBPHFaceRecognizer_create()
'''
static Ptr< LBPHFaceRecognizer > create(int radius = 1,
int 	neighbors = 8,
int 	grid_x = 8,
int 	grid_y = 8,
double 	threshold = DBL_MAX 
)		'''
#Parameters
'''
radius->	The radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get.
neighbors->	The number of sample points to build a Circular Local Binary Pattern from. An appropriate value is to use 8 sample points. Keep in mind: the more sample points you include, the higher the computational cost.
grid_x->	The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
grid_y->	The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
threshold->	The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.'''

