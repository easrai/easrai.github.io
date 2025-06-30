---
layout: post
title: numpy arrays
date: 2025-05-26 14:24:00
description: notes about numpy arrays
tags: notes
categories: notes
giscus_comments: True
---


NumPy, short for Numerical Python, is one of the most important foundational packages for numerical computing in Python.
Most computational packages providing scientific functionality use NumPy's array objects as the *lingua franca* for data exchange

Some of the things in NumPy:
- ndarray, an efficient multidimensional array providing fast array-oriented arithmetic operations and flexible broadcasting capabilities.
- Mathematical functions for fast operations on entire arrays of data without having to write loops.
- Tools for reading/writing array data to disk and working with memory mapped files.
- Linear algebra, random number generation, and Fourier transform capabilities.
- A C API for connecting NumPy with libraries written in C, C++ or FORTRAN.
## The NumPy ndarray: A Multidimensional Array Object

N-dimensional array object or ndarray is a fast, flexible container for large datsets in Python.
They enable us to perform mathematical operations on whole blocks of data using similar syntax to the equivalent 

```Python
In[12]: import numpy as np
#Generate some random data
In[13]:data = np.random.module.randn(2,3)

In[14]:data
Out[14]:
array([[-0.2047, 0.4789, -0.5194],
		[-0.5557, 1.9658, 1.3934]])
```

Then i can do mathematical operations with data like `data + data` and `data * 10` etc.
An ndarray is a generic multidimensional container for homogenous data that, all of the elements must be of the **same type**.

Every array has a *shape* indicating size of each dimension and *dtype*, an object describing *data type* of the array:
```Python
In [17]:data.shape
Out[17]: (2,3)

In [18]:data.dtype
Out[18]:dtype('float64')
```
This chapter will introduce you to the basics of using NumPy arrays adn should be sufficient for following along with the rest of the book

## Creating ndarrays

The easiest way to create an array is using the `array` function. This accepts any sequence-like object (including other arrays) and produces a new NumPy array containing the passed data.
For example, a list is a good candidate for conversion:
```Python
In [19]: data1 = [6, 7.5, 8, 0, 1]

In [20]: arr1 = np.array(data1)

In [21]: arr1
Out[21]:array([ 6. , 7.5, 8. , 1. ])
```
Nested sequences like a list of equal length lists, will be converted into a multidimensional array:
```Python
In [22]: data2 = [[1, 2, 3, 4],[5, 6, 7, 8]]

In [23]: arr2 = np.array(data2)

In [24]: arr2
Out[24]:
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
```
`.ndim` method returns the length of array.

Unless explicitly specified `np.array` tries to infer a good data type for the array that it creates. The data type is stored in a special `dtype` metadata object; for example, in the previous two examples we have:
```Python
In [27]: arr1.dtype
Out[27]: dtype('float64')

In [28]: arr2.dtype
Out[28]: dtype('int64')
```
In addition to `np.array`, there are a number of other functions for creating new arrays. As examples, `zeros` and `ones` create arrays of 0s and 1s respectively with a given length or shape
`empty` creates an array without initializing its values to any particular value.
To create a higher dimensional array with these methods pass a tuple for the shape

## Data Types for ndarrays
The *data type* or `dtype` is  a special object containing the information (or *metadata*, data about data) the ndarray needs to interpret a chunk of memory as a particular type of data:
```Python
In [33]: arr1 = np.array([1, 2, 3], dtype=np.float64)

In [34]: arr2 = np.array([1,2,3], dtype=np.int32)

In [35]: arr1.dtype
Out[35]: dtype('float64')
In [36]: arr2.dtype
Out[36]: dtype('int32')
```

dtypes are a source of NumPy's flexibility for interacting with data coming from other systems.

You can explaicitly convert or *cast* an array from one dtype to another using ndarray's `astype` method:
```Python
In [37]: arr = np.array([1, 2, 3, 4, 5])

In [38]: arr.dtype
Out[38]: dtype('int64')

In [39]: float_arr = arr.astype(np.float64)

In [40]: float_arr.dtype
Out[40]: dtype('float64')
```
In this example, integers were cast to floating point. If I cast some floating-point numbers to be of integer dtype, the decimal part will be truncated:
```Python
In [41]: arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

In [42]: arr
Out[42]: array([ 3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

In [43]: arr.astype(np.int32)
Out[43]: array([ 3, -1, -2, 0, 12, 10], dtype=int32)

In [44]: numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

In [45]: numeric_strings.astype(float)
Out[45]: array([ 1.25, -9.6 , 42. ])
```

### Warning
It’s important to be cautious when using the `numpy.string_ type`,
as string data in NumPy is fixed size and may truncate input
without warning. pandas has more intuitive out-of-the-box behavior on non-numeric data.

If casting were to fail for some reason, `ValueError` will be raised.

## Arithmetic with NumPy Arrays

Arrays are important because they enable you to express batch operations on data without writing any `for` loops. `NumPy` users call this *vectorization.*
Any arithmetic operations between equal-size arrays applies the operation element-wise:

```Python
In [51]: arr = np.array([[1., 2., 3.],[4., 5., 6.]])

In [52]: arr
Out[52]:
array([[ 1., 2., 3.],
       [ 4., 5., 6.]])

In [53]: arr*arr
Out[53]:
array([[  1.,  4.,  9.],
       [ 16., 25., 36.]])

In [54]: arr - arr
Out[54]:
array([[ 0., 0., 0.],
       [ 0., 0., 0.]])

```

Arithmetic operations with scalars propagate the scalar argument to each element in the array.
Comparison between array of the same size yield boolean arrays

```Python
In [55]: 1 / arr
Out[55]:
array([[ 1. , 0.5 , 0.3333],
[ 0.25 , 0.2 , 0.1667]])

In [56]: arr ** 0.5
Out[56]:
array([[ 1. , 1.4142, 1.7321],
[ 2. , 2.2361, 2.4495]])

In [57]: arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])

In [58]: arr2
Out[58]:
array([[ 0., 4., 1.],
[ 7., 2., 12.]])

In [59]: arr2 > arr
Out[59]:
array([[False, True, False],
[ True, False, True]], dtype=bool)

```

Operations between differently sized arrays is called **broadcasting**.


## Basic indexing and slicing

NumPy array indexing is a rich topic, as there are many ways you may want to select a subset of your data or individual elements.

```Python
In [60]: arr = np.arange(10)

In [61]: arr
Out[61]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

In [62]: arr[5]
Out[62]: 5

In [63]: arr[5:8]
Out[63]: array([5, 6, 7])

In [64]: arr[5:8] = 12

In [65]: arr
Out[65]: array([ 0, 1, 2, 3, 4, 12, 12, 12, 8, 9])

```
If you assign a scalar value to a slice, as in `arr[5:8] = 12` the value is propagated (or *broadcasted* henceforth) to the entire selection.

Array slices are *views* on the original array, which means
1. Data is not copied
2. Any change to the view (like assignment) is reflected in the original array

>
>If you want copy of a slice of an array, then you need to explicitly copy the array, like `arr[5:8].copy()`

 
 
 An example(to indicate that slices are indeed views on the original array)
 ```Python
In [66]: arr_slice = arr[5:8]

In [67]: arr_slice
Out[67]: array([12, 12, 12])

In [68]: arr_slice[1] = 12345

In [69]: arr
Out[69]: array([ 0, 1, 2, 3, 4, 12, 12345, 12, 8,
9])

In [70]: arr_slice[:] = 64

In [71]: arr
Out[71]: array([ 0, 1, 2, 3, 4, 64, 64, 64, 8, 9])

 ```

In higher dimensional arrays we have many more option, like in two dimensional arrays the elements at each index are no longer scalar but one dimensional arrays, so individual elements of these arrays can be accessed via double indexing.


Individual elements in two dimensional arrays can be accessed recursively but that's a bit of work. So, a better option is to pass tuple of two elements like below

```Python
In [74]: arr2d[0][2]
Out[74]: 3

In [75]: arr2d[0, 2]
Out[75]: 3
```

In multidimensional arrays, if you omit later indices, the returned object will be a lower dimensional ndarray consisting of all the data along the higher dimensions.

## Indexing with slices

Like one dimensional objects such as python lists, ndarrays can be sliced with the familiar syntax:

```Python
In [88]: arr
Out[88]: array([ 0, 1, 2, 3, 4, 64, 64, 64, 8, 9])

In [89]: arr[1:6]
Out[89]: array([ 1, 2, 3, 4, 64])
```

You can also pass multiple indices like in case you do with python lists.

## Boolean Indexing

Let's consider an example where we have some data in an array and an array of names with duplicates. I'm going to use here the `randn` function in `numpy.random` to generate some random normally distributed data:

```Python
In [98]: names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

In [99]: data = np.random.randn(7, 4)

In [100]: names
Out[100]:
array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'],
	dtype='<U4')

In [101]: data
Out[101]:
array([[ 0.0929, 0.2817, 0.769 , 1.2464],
       [ 1.0072, -1.2962, 0.275 , 0.2289],
       [ 1.3529, 0.8864, -2.0016, -0.3718],
       [ 1.669 , -0.4386, -0.5397, 0.477 ],
       [ 3.2489, -1.0212, -0.5771, 0.1241],
       [ 0.3026, 0.5238, 0.0009, 1.3438],
       [-0.7135, -0.8312, -2.3702, -1.8608]])
```

The boolean condition `names == 'Bob'` returns a boolean array, whose *i*th index element is `names[i] == 'Bob'`

This array can be further passed to `data` , which gives array containing elements in whose index the boolean array is true, which means that the length of both `data` and `names` must be same.

>
>Boolean selection will not fail if the boolean array is not the correct length, so it must be used with care.

You can further slice the array `data[names == 'Bob']` the way you want and it will give suitable results. To be clear you can treat it like any other ndarray. 

The above idea can be further extended to the definition of `arr[bool]` where `arr[bool][i] = True` if `bool[i]` is and `arr[bool][i] = False` if `bool[i]` is `False`.

>
>Despite this extension, it is more safe to use &(and) and |(or) instead of and and or, this is because boolean arrays only work with these symbols

Setting values with boolean arrays is also, easy as boolean arrays are in essence subarray of an array, a single assignment would change the value of all the elements for whose index the boolean condition is true.

>
>Here boolean array refers to something like `arr[bool]`

## Fancy Indexing

*Fancy Indexing* is a term adopted by NumPy to describe indexing using integer arrays.
Suppose we had an 8 x 4 array:

```Python
In [117]: arr = np.empty((8, 4))
In [118]: for i in range(8):
	    :    arr[i] = i
In [119]: arr
Out[119]:
array([[ 0., 0., 0., 0.],
       [ 1., 1., 1., 1.],
       [ 2., 2., 2., 2.],
       [ 3., 3., 3., 3.],
       [ 4., 4., 4., 4.],
       [ 5., 5., 5., 5.],
       [ 6., 6., 6., 6.],
       [ 7., 7., 7., 7.]])
```

To select out a subset of rows you can simply pass a list or ndarray containing those specific indexes.

```Python
In [120]: arr[[4, 3, 0, 6]]
Out[120]:
array([[ 4., 4., 4., 4.],
       [ 3., 3., 3., 3.],
       [ 0., 0., 0., 0.],
       [ 6., 6., 6., 6.]])
```

```Python
In [122]: arr = np.arange(32).reshape((8, 4))

In [123]: arr
Out[123]:
array([[ 0, 1, 2, 3],
       [ 4, 5, 6, 7],
       [ 8, 9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
In [124]: arr[[1, 5, 7, 2], [0, 3, 1, 2]]
Out[124]: array([ 4, 23, 29, 10])
```

Here `np.arange` creates an array of integers from 1 to 32 in ascending order, and `.reshape` method just creates an array which as no. of elements equal to the first argument and each element is an array having cardinal number equal to the second argument.

Fancy indexing unlike slicing always copies the data into a new array meaning any change in the resultant fancy indexed array does not change the parent array.

## Transposing array and swapping Axes

Transposing is a special form of reshaping that similarly returns a view on the underlying data without copying anything. Arrays have the `transpose` method and also the special T attribute:

```Python
In [126]: arr = np.arange(15).reshape((3, 5))
In [127]: arr
Out[127]:
array([[ 0, 1, 2, 3, 4],
       [ 5, 6, 7, 8, 9],
       [10, 11, 12, 13, 14]])
In [128]: arr.T
Out[128]:
array([[ 0, 5, 10],
       [ 1, 6, 11],
       [ 2, 7, 12],
       [ 3, 8, 13],
       [ 4, 9, 14]])
```

When doing matrix computation, we can compute the matrix inner product using `np.dot(matrix1,matrix2)` where `matrix1` and `matrix2` are numpy arrays.

For higher dimensional arrays, `transpose` will accept a tuple of axis numbers to permute the axes (for extra mind bending):
```Python
In [132]: arr = np.arange(16).reshape((2,2,4))

In [133]: arr
Out[133]:
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]]])

In [134]: arr.transpose((1, 0, 2))
Out[134]:
array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],
       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
```

>
>This is how `transpose` function or method works:
>Suppose the input array is $n$ dimensional and the input tuple of `transpose` method is essentially a permutation of numbers from 0 to $(n-1)$. Let the input permutation be $(a_1, a_2 , a_3 , ..., a_n)$ and the initial array be `arr` and new array be `narr`, then what we are doing is essentially the assignments,
```python
     narr [m1] [m2] [m3] [m4] .... [mn] = arr [ma1]  [ma2]  [ma3] .... [man]
```

Similarly there is `swapaxes` method in python, this method works by returning a view on data without making a copy.

This method accepts two values. the axes that need to be swapped, it is essentially a transposition, which means it is equivalent to applying `transpose` method to the permutation generated by this transposition.

## Universal Functions: Fast Element-Wise Array Functions

A universal function or *ufunc*, is a function that performs element-wise operations on data in ndarrays. They can be thought of as fast vectorized wrappers for simple functions that take one or more scalar values and produce one or more scalar results.

Many ufuncs are simple element- wise transformations, like `sqrt` or `exp`.

`np.sqrt(array)` just takes the array as input and generates(a copy) an array with each element square rooted.

Such ufuncs which take only element as input are *unary* ufuncs and others such as `add` and `maximum`, take two arrays (thus, *binary* ufuncs) and return a single array as result.

Both the above methods generate an array, thus its obvious to accurately guess what they do.

**Not common,** but ufunc can return multiple arrays. `modf` is one example, a vectorized version of built-in Python `divmod`; it returns the fractional and integral parts of a floating-point array:

It returns the tuple `(remainder, whole part)`

Ufuncs also accept an optional `out` argument that allows them to operate in place on arrays.
 >
 >`out` is the second argument,  applying the ufunc with this argument, just assigns the resulting array to the variable in `out`
  



