import tensorflow as tf
import tempfile
'''
Datasets

You can use Python iteration over the tf.data.Dataset object 
and do not need to explicitly create an tf.data.Iterator object.
'''

tf.enable_eager_execution()
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])

#create a CSV file

_ , filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""
Line 2
Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)

# Apply transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x,end='\n')

print('\nElements in ds_file:')
for x in ds_file:
    print(x)









