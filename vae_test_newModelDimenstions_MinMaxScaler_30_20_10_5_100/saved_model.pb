??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
 autoencoder/encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" autoencoder/encoder/dense/kernel
?
4autoencoder/encoder/dense/kernel/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense/kernel*
_output_shapes

:*
dtype0
?
autoencoder/encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name autoencoder/encoder/dense/bias
?
2autoencoder/encoder/dense/bias/Read/ReadVariableOpReadVariableOpautoencoder/encoder/dense/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"autoencoder/encoder/dense_1/kernel
?
6autoencoder/encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_1/kernel*
_output_shapes

:*
dtype0
?
 autoencoder/encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_1/bias
?
4autoencoder/encoder/dense_1/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_1/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"autoencoder/encoder/dense_2/kernel
?
6autoencoder/encoder/dense_2/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_2/kernel*
_output_shapes

:*
dtype0
?
 autoencoder/encoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_2/bias
?
4autoencoder/encoder/dense_2/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_2/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"autoencoder/encoder/dense_3/kernel
?
6autoencoder/encoder/dense_3/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_3/kernel*
_output_shapes

:*
dtype0
?
 autoencoder/encoder/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_3/bias
?
4autoencoder/encoder/dense_3/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_3/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*3
shared_name$"autoencoder/encoder/dense_4/kernel
?
6autoencoder/encoder/dense_4/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_4/kernel*
_output_shapes

:
*
dtype0
?
 autoencoder/encoder/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" autoencoder/encoder/dense_4/bias
?
4autoencoder/encoder/dense_4/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_4/bias*
_output_shapes
:
*
dtype0
?
"autoencoder/encoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*3
shared_name$"autoencoder/encoder/dense_5/kernel
?
6autoencoder/encoder/dense_5/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_5/kernel*
_output_shapes

:
*
dtype0
?
 autoencoder/encoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_5/bias
?
4autoencoder/encoder/dense_5/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_5/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*3
shared_name$"autoencoder/encoder/dense_6/kernel
?
6autoencoder/encoder/dense_6/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_6/kernel*
_output_shapes

:
*
dtype0
?
 autoencoder/encoder/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_6/bias
?
4autoencoder/encoder/dense_6/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_6/bias*
_output_shapes
:*
dtype0
?
"autoencoder/decoder/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*3
shared_name$"autoencoder/decoder/dense_7/kernel
?
6autoencoder/decoder/dense_7/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/decoder/dense_7/kernel*
_output_shapes

:
*
dtype0
?
 autoencoder/decoder/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" autoencoder/decoder/dense_7/bias
?
4autoencoder/decoder/dense_7/bias/Read/ReadVariableOpReadVariableOp autoencoder/decoder/dense_7/bias*
_output_shapes
:
*
dtype0
?
"autoencoder/decoder/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*3
shared_name$"autoencoder/decoder/dense_8/kernel
?
6autoencoder/decoder/dense_8/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/decoder/dense_8/kernel*
_output_shapes

:
*
dtype0
?
 autoencoder/decoder/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/decoder/dense_8/bias
?
4autoencoder/decoder/dense_8/bias/Read/ReadVariableOpReadVariableOp autoencoder/decoder/dense_8/bias*
_output_shapes
:*
dtype0
?
"autoencoder/decoder/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"autoencoder/decoder/dense_9/kernel
?
6autoencoder/decoder/dense_9/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/decoder/dense_9/kernel*
_output_shapes

:*
dtype0
?
 autoencoder/decoder/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/decoder/dense_9/bias
?
4autoencoder/decoder/dense_9/bias/Read/ReadVariableOpReadVariableOp autoencoder/decoder/dense_9/bias*
_output_shapes
:*
dtype0
?
#autoencoder/decoder/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#autoencoder/decoder/dense_10/kernel
?
7autoencoder/decoder/dense_10/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/decoder/dense_10/kernel*
_output_shapes

:*
dtype0
?
!autoencoder/decoder/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!autoencoder/decoder/dense_10/bias
?
5autoencoder/decoder/dense_10/bias/Read/ReadVariableOpReadVariableOp!autoencoder/decoder/dense_10/bias*
_output_shapes
:*
dtype0
?
#autoencoder/decoder/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#autoencoder/decoder/dense_11/kernel
?
7autoencoder/decoder/dense_11/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/decoder/dense_11/kernel*
_output_shapes

:*
dtype0
?
!autoencoder/decoder/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!autoencoder/decoder/dense_11/bias
?
5autoencoder/decoder/dense_11/bias/Read/ReadVariableOpReadVariableOp!autoencoder/decoder/dense_11/bias*
_output_shapes
:*
dtype0
?
#autoencoder/decoder/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#autoencoder/decoder/dense_12/kernel
?
7autoencoder/decoder/dense_12/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/decoder/dense_12/kernel*
_output_shapes

:*
dtype0
?
!autoencoder/decoder/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!autoencoder/decoder/dense_12/bias
?
5autoencoder/decoder/dense_12/bias/Read/ReadVariableOpReadVariableOp!autoencoder/decoder/dense_12/bias*
_output_shapes
:*
dtype0
?
'Adam/autoencoder/encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/autoencoder/encoder/dense/kernel/m
?
;Adam/autoencoder/encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense/kernel/m*
_output_shapes

:*
dtype0
?
%Adam/autoencoder/encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/autoencoder/encoder/dense/bias/m
?
9Adam/autoencoder/encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp%Adam/autoencoder/encoder/dense/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_1/kernel/m
?
=Adam/autoencoder/encoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_1/bias/m
?
;Adam/autoencoder/encoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_1/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_2/kernel/m
?
=Adam/autoencoder/encoder/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_2/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_2/bias/m
?
;Adam/autoencoder/encoder/dense_2/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_2/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_3/kernel/m
?
=Adam/autoencoder/encoder/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_3/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_3/bias/m
?
;Adam/autoencoder/encoder/dense_3/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_3/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_4/kernel/m
?
=Adam/autoencoder/encoder/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_4/kernel/m*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/autoencoder/encoder/dense_4/bias/m
?
;Adam/autoencoder/encoder/dense_4/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_4/bias/m*
_output_shapes
:
*
dtype0
?
)Adam/autoencoder/encoder/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_5/kernel/m
?
=Adam/autoencoder/encoder/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_5/kernel/m*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_5/bias/m
?
;Adam/autoencoder/encoder/dense_5/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_5/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_6/kernel/m
?
=Adam/autoencoder/encoder/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_6/kernel/m*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_6/bias/m
?
;Adam/autoencoder/encoder/dense_6/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_6/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/decoder/dense_7/kernel/m
?
=Adam/autoencoder/decoder/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_7/kernel/m*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/decoder/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/autoencoder/decoder/dense_7/bias/m
?
;Adam/autoencoder/decoder/dense_7/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_7/bias/m*
_output_shapes
:
*
dtype0
?
)Adam/autoencoder/decoder/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/decoder/dense_8/kernel/m
?
=Adam/autoencoder/decoder/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_8/kernel/m*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/decoder/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/decoder/dense_8/bias/m
?
;Adam/autoencoder/decoder/dense_8/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_8/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/decoder/dense_9/kernel/m
?
=Adam/autoencoder/decoder/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_9/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/decoder/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/decoder/dense_9/bias/m
?
;Adam/autoencoder/decoder/dense_9/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_9/bias/m*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_10/kernel/m
?
>Adam/autoencoder/decoder/dense_10/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_10/kernel/m*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_10/bias/m
?
<Adam/autoencoder/decoder/dense_10/bias/m/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_10/bias/m*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_11/kernel/m
?
>Adam/autoencoder/decoder/dense_11/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_11/kernel/m*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_11/bias/m
?
<Adam/autoencoder/decoder/dense_11/bias/m/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_11/bias/m*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_12/kernel/m
?
>Adam/autoencoder/decoder/dense_12/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_12/kernel/m*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_12/bias/m
?
<Adam/autoencoder/decoder/dense_12/bias/m/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_12/bias/m*
_output_shapes
:*
dtype0
?
'Adam/autoencoder/encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/autoencoder/encoder/dense/kernel/v
?
;Adam/autoencoder/encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense/kernel/v*
_output_shapes

:*
dtype0
?
%Adam/autoencoder/encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/autoencoder/encoder/dense/bias/v
?
9Adam/autoencoder/encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp%Adam/autoencoder/encoder/dense/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_1/kernel/v
?
=Adam/autoencoder/encoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_1/bias/v
?
;Adam/autoencoder/encoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_1/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_2/kernel/v
?
=Adam/autoencoder/encoder/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_2/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_2/bias/v
?
;Adam/autoencoder/encoder/dense_2/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_2/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/encoder/dense_3/kernel/v
?
=Adam/autoencoder/encoder/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_3/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/encoder/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_3/bias/v
?
;Adam/autoencoder/encoder/dense_3/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_3/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_4/kernel/v
?
=Adam/autoencoder/encoder/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_4/kernel/v*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/autoencoder/encoder/dense_4/bias/v
?
;Adam/autoencoder/encoder/dense_4/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_4/bias/v*
_output_shapes
:
*
dtype0
?
)Adam/autoencoder/encoder/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_5/kernel/v
?
=Adam/autoencoder/encoder/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_5/kernel/v*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_5/bias/v
?
;Adam/autoencoder/encoder/dense_5/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_5/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/encoder/dense_6/kernel/v
?
=Adam/autoencoder/encoder/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_6/kernel/v*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/encoder/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_6/bias/v
?
;Adam/autoencoder/encoder/dense_6/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_6/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/decoder/dense_7/kernel/v
?
=Adam/autoencoder/decoder/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_7/kernel/v*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/decoder/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/autoencoder/decoder/dense_7/bias/v
?
;Adam/autoencoder/decoder/dense_7/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_7/bias/v*
_output_shapes
:
*
dtype0
?
)Adam/autoencoder/decoder/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*:
shared_name+)Adam/autoencoder/decoder/dense_8/kernel/v
?
=Adam/autoencoder/decoder/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_8/kernel/v*
_output_shapes

:
*
dtype0
?
'Adam/autoencoder/decoder/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/decoder/dense_8/bias/v
?
;Adam/autoencoder/decoder/dense_8/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_8/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/autoencoder/decoder/dense_9/kernel/v
?
=Adam/autoencoder/decoder/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_9/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/autoencoder/decoder/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/decoder/dense_9/bias/v
?
;Adam/autoencoder/decoder/dense_9/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_9/bias/v*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_10/kernel/v
?
>Adam/autoencoder/decoder/dense_10/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_10/kernel/v*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_10/bias/v
?
<Adam/autoencoder/decoder/dense_10/bias/v/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_10/bias/v*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_11/kernel/v
?
>Adam/autoencoder/decoder/dense_11/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_11/kernel/v*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_11/bias/v
?
<Adam/autoencoder/decoder/dense_11/bias/v/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_11/bias/v*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/decoder/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/autoencoder/decoder/dense_12/kernel/v
?
>Adam/autoencoder/decoder/dense_12/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/decoder/dense_12/kernel/v*
_output_shapes

:*
dtype0
?
(Adam/autoencoder/decoder/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/decoder/dense_12/bias/v
?
<Adam/autoencoder/decoder/dense_12/bias/v/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/decoder/dense_12/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*֛
value˛BǛ B??
?
encoder
decoder
	optimizer

signatures
	variables
regularization_losses
	keras_api
trainable_variables
?
	encoder_layer1

encoder_active_layer1
encoder_layer2
encoder_active_layer2
encoder_layer3
encoder_active_layer3
encoder_layer4
encoder_active_layer4
encoder_layer5
encoder_active_layer5

dense_mean
dense_log_var
sampling
	variables
regularization_losses
	keras_api
trainable_variables
?
decoder_layer1
decoder_active_layer1
decoder_layer2
decoder_active_layer2
decoder_layer3
decoder_active_layer3
 decoder_layer4
!decoder_active_layer4
"decoder_layer5
#decoder_active_layer5
$dense_output
%decoder_active_output
&	variables
'regularization_losses
(	keras_api
)trainable_variables
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?
 
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25
 
?

Ilayers
trainable_variables
	variables
Jlayer_regularization_losses
Kmetrics
regularization_losses
Lnon_trainable_variables
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25
h

/kernel
0bias
M	variables
Nregularization_losses
O	keras_api
Ptrainable_variables
R
Q	variables
Rregularization_losses
S	keras_api
Ttrainable_variables
h

1kernel
2bias
U	variables
Vregularization_losses
W	keras_api
Xtrainable_variables
R
Y	variables
Zregularization_losses
[	keras_api
\trainable_variables
h

3kernel
4bias
]	variables
^regularization_losses
_	keras_api
`trainable_variables
R
a	variables
bregularization_losses
c	keras_api
dtrainable_variables
h

5kernel
6bias
e	variables
fregularization_losses
g	keras_api
htrainable_variables
R
i	variables
jregularization_losses
k	keras_api
ltrainable_variables
h

7kernel
8bias
m	variables
nregularization_losses
o	keras_api
ptrainable_variables
R
q	variables
rregularization_losses
s	keras_api
ttrainable_variables
h

9kernel
:bias
u	variables
vregularization_losses
w	keras_api
xtrainable_variables
h

;kernel
<bias
y	variables
zregularization_losses
{	keras_api
|trainable_variables
S
}	variables
~regularization_losses
	keras_api
?trainable_variables
f
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
 
?
?layers
trainable_variables
	variables
 ?layer_regularization_losses
?metrics
regularization_losses
?non_trainable_variables
f
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
l

=kernel
>bias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
l

?kernel
@bias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
l

Akernel
Bbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
l

Ckernel
Dbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
l

Ekernel
Fbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
l

Gkernel
Hbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
?	variables
?regularization_losses
?	keras_api
?trainable_variables
V
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
 
?
?layers
)trainable_variables
&	variables
 ?layer_regularization_losses
?metrics
'regularization_losses
?non_trainable_variables
V
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEautoencoder/encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"autoencoder/encoder/dense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE autoencoder/encoder/dense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"autoencoder/encoder/dense_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE autoencoder/encoder/dense_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"autoencoder/decoder/dense_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE autoencoder/decoder/dense_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"autoencoder/decoder/dense_8/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE autoencoder/decoder/dense_8/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"autoencoder/decoder/dense_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE autoencoder/decoder/dense_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#autoencoder/decoder/dense_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!autoencoder/decoder/dense_10/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#autoencoder/decoder/dense_11/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!autoencoder/decoder/dense_11/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#autoencoder/decoder/dense_12/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!autoencoder/decoder/dense_12/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 

/0
01
 
?
?layers
Ptrainable_variables
M	variables
 ?layer_regularization_losses
?metrics
Nregularization_losses
?non_trainable_variables

/0
01
 
 
?
?layers
Ttrainable_variables
Q	variables
 ?layer_regularization_losses
?metrics
Rregularization_losses
?non_trainable_variables
 

10
21
 
?
?layers
Xtrainable_variables
U	variables
 ?layer_regularization_losses
?metrics
Vregularization_losses
?non_trainable_variables

10
21
 
 
?
?layers
\trainable_variables
Y	variables
 ?layer_regularization_losses
?metrics
Zregularization_losses
?non_trainable_variables
 

30
41
 
?
?layers
`trainable_variables
]	variables
 ?layer_regularization_losses
?metrics
^regularization_losses
?non_trainable_variables

30
41
 
 
?
?layers
dtrainable_variables
a	variables
 ?layer_regularization_losses
?metrics
bregularization_losses
?non_trainable_variables
 

50
61
 
?
?layers
htrainable_variables
e	variables
 ?layer_regularization_losses
?metrics
fregularization_losses
?non_trainable_variables

50
61
 
 
?
?layers
ltrainable_variables
i	variables
 ?layer_regularization_losses
?metrics
jregularization_losses
?non_trainable_variables
 

70
81
 
?
?layers
ptrainable_variables
m	variables
 ?layer_regularization_losses
?metrics
nregularization_losses
?non_trainable_variables

70
81
 
 
?
?layers
ttrainable_variables
q	variables
 ?layer_regularization_losses
?metrics
rregularization_losses
?non_trainable_variables
 

90
:1
 
?
?layers
xtrainable_variables
u	variables
 ?layer_regularization_losses
?metrics
vregularization_losses
?non_trainable_variables

90
:1

;0
<1
 
?
?layers
|trainable_variables
y	variables
 ?layer_regularization_losses
?metrics
zregularization_losses
?non_trainable_variables

;0
<1
 
 
?
?layers
?trainable_variables
}	variables
 ?layer_regularization_losses
?metrics
~regularization_losses
?non_trainable_variables
 
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
 
 
 

=0
>1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

=0
>1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 

?0
@1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

?0
@1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 

A0
B1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

A0
B1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 

C0
D1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

C0
D1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 

E0
F1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

E0
F1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 

G0
H1
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables

G0
H1
 
 
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
 
V
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/autoencoder/encoder/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_7/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_7/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_8/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_8/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_9/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_9/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_10/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_10/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_11/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_11/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_12/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_12/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/autoencoder/encoder/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_7/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_7/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_8/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_8/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_9/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_9/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_10/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_10/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_11/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_11/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/decoder/dense_12/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/autoencoder/decoder/dense_12/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 autoencoder/encoder/dense/kernelautoencoder/encoder/dense/bias"autoencoder/encoder/dense_1/kernel autoencoder/encoder/dense_1/bias"autoencoder/encoder/dense_2/kernel autoencoder/encoder/dense_2/bias"autoencoder/encoder/dense_3/kernel autoencoder/encoder/dense_3/bias"autoencoder/encoder/dense_4/kernel autoencoder/encoder/dense_4/bias"autoencoder/encoder/dense_5/kernel autoencoder/encoder/dense_5/bias"autoencoder/encoder/dense_6/kernel autoencoder/encoder/dense_6/bias"autoencoder/decoder/dense_7/kernel autoencoder/decoder/dense_7/bias"autoencoder/decoder/dense_8/kernel autoencoder/decoder/dense_8/bias"autoencoder/decoder/dense_9/kernel autoencoder/decoder/dense_9/bias#autoencoder/decoder/dense_10/kernel!autoencoder/decoder/dense_10/bias#autoencoder/decoder/dense_11/kernel!autoencoder/decoder/dense_11/bias#autoencoder/decoder/dense_12/kernel!autoencoder/decoder/dense_12/bias*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_4606226
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp4autoencoder/encoder/dense/kernel/Read/ReadVariableOp2autoencoder/encoder/dense/bias/Read/ReadVariableOp6autoencoder/encoder/dense_1/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_1/bias/Read/ReadVariableOp6autoencoder/encoder/dense_2/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_2/bias/Read/ReadVariableOp6autoencoder/encoder/dense_3/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_3/bias/Read/ReadVariableOp6autoencoder/encoder/dense_4/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_4/bias/Read/ReadVariableOp6autoencoder/encoder/dense_5/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_5/bias/Read/ReadVariableOp6autoencoder/encoder/dense_6/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_6/bias/Read/ReadVariableOp6autoencoder/decoder/dense_7/kernel/Read/ReadVariableOp4autoencoder/decoder/dense_7/bias/Read/ReadVariableOp6autoencoder/decoder/dense_8/kernel/Read/ReadVariableOp4autoencoder/decoder/dense_8/bias/Read/ReadVariableOp6autoencoder/decoder/dense_9/kernel/Read/ReadVariableOp4autoencoder/decoder/dense_9/bias/Read/ReadVariableOp7autoencoder/decoder/dense_10/kernel/Read/ReadVariableOp5autoencoder/decoder/dense_10/bias/Read/ReadVariableOp7autoencoder/decoder/dense_11/kernel/Read/ReadVariableOp5autoencoder/decoder/dense_11/bias/Read/ReadVariableOp7autoencoder/decoder/dense_12/kernel/Read/ReadVariableOp5autoencoder/decoder/dense_12/bias/Read/ReadVariableOp;Adam/autoencoder/encoder/dense/kernel/m/Read/ReadVariableOp9Adam/autoencoder/encoder/dense/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_1/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_1/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_2/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_2/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_3/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_3/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_4/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_4/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_5/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_5/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_6/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_6/bias/m/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_7/kernel/m/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_7/bias/m/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_8/kernel/m/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_8/bias/m/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_9/kernel/m/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_9/bias/m/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_10/kernel/m/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_10/bias/m/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_11/kernel/m/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_11/bias/m/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_12/kernel/m/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_12/bias/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense/kernel/v/Read/ReadVariableOp9Adam/autoencoder/encoder/dense/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_1/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_1/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_2/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_2/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_3/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_3/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_4/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_4/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_5/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_5/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_6/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_6/bias/v/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_7/kernel/v/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_7/bias/v/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_8/kernel/v/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_8/bias/v/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_9/kernel/v/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_9/bias/v/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_10/kernel/v/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_10/bias/v/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_11/kernel/v/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_11/bias/v/Read/ReadVariableOp>Adam/autoencoder/decoder/dense_12/kernel/v/Read/ReadVariableOp<Adam/autoencoder/decoder/dense_12/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_4606659
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate autoencoder/encoder/dense/kernelautoencoder/encoder/dense/bias"autoencoder/encoder/dense_1/kernel autoencoder/encoder/dense_1/bias"autoencoder/encoder/dense_2/kernel autoencoder/encoder/dense_2/bias"autoencoder/encoder/dense_3/kernel autoencoder/encoder/dense_3/bias"autoencoder/encoder/dense_4/kernel autoencoder/encoder/dense_4/bias"autoencoder/encoder/dense_5/kernel autoencoder/encoder/dense_5/bias"autoencoder/encoder/dense_6/kernel autoencoder/encoder/dense_6/bias"autoencoder/decoder/dense_7/kernel autoencoder/decoder/dense_7/bias"autoencoder/decoder/dense_8/kernel autoencoder/decoder/dense_8/bias"autoencoder/decoder/dense_9/kernel autoencoder/decoder/dense_9/bias#autoencoder/decoder/dense_10/kernel!autoencoder/decoder/dense_10/bias#autoencoder/decoder/dense_11/kernel!autoencoder/decoder/dense_11/bias#autoencoder/decoder/dense_12/kernel!autoencoder/decoder/dense_12/bias'Adam/autoencoder/encoder/dense/kernel/m%Adam/autoencoder/encoder/dense/bias/m)Adam/autoencoder/encoder/dense_1/kernel/m'Adam/autoencoder/encoder/dense_1/bias/m)Adam/autoencoder/encoder/dense_2/kernel/m'Adam/autoencoder/encoder/dense_2/bias/m)Adam/autoencoder/encoder/dense_3/kernel/m'Adam/autoencoder/encoder/dense_3/bias/m)Adam/autoencoder/encoder/dense_4/kernel/m'Adam/autoencoder/encoder/dense_4/bias/m)Adam/autoencoder/encoder/dense_5/kernel/m'Adam/autoencoder/encoder/dense_5/bias/m)Adam/autoencoder/encoder/dense_6/kernel/m'Adam/autoencoder/encoder/dense_6/bias/m)Adam/autoencoder/decoder/dense_7/kernel/m'Adam/autoencoder/decoder/dense_7/bias/m)Adam/autoencoder/decoder/dense_8/kernel/m'Adam/autoencoder/decoder/dense_8/bias/m)Adam/autoencoder/decoder/dense_9/kernel/m'Adam/autoencoder/decoder/dense_9/bias/m*Adam/autoencoder/decoder/dense_10/kernel/m(Adam/autoencoder/decoder/dense_10/bias/m*Adam/autoencoder/decoder/dense_11/kernel/m(Adam/autoencoder/decoder/dense_11/bias/m*Adam/autoencoder/decoder/dense_12/kernel/m(Adam/autoencoder/decoder/dense_12/bias/m'Adam/autoencoder/encoder/dense/kernel/v%Adam/autoencoder/encoder/dense/bias/v)Adam/autoencoder/encoder/dense_1/kernel/v'Adam/autoencoder/encoder/dense_1/bias/v)Adam/autoencoder/encoder/dense_2/kernel/v'Adam/autoencoder/encoder/dense_2/bias/v)Adam/autoencoder/encoder/dense_3/kernel/v'Adam/autoencoder/encoder/dense_3/bias/v)Adam/autoencoder/encoder/dense_4/kernel/v'Adam/autoencoder/encoder/dense_4/bias/v)Adam/autoencoder/encoder/dense_5/kernel/v'Adam/autoencoder/encoder/dense_5/bias/v)Adam/autoencoder/encoder/dense_6/kernel/v'Adam/autoencoder/encoder/dense_6/bias/v)Adam/autoencoder/decoder/dense_7/kernel/v'Adam/autoencoder/decoder/dense_7/bias/v)Adam/autoencoder/decoder/dense_8/kernel/v'Adam/autoencoder/decoder/dense_8/bias/v)Adam/autoencoder/decoder/dense_9/kernel/v'Adam/autoencoder/decoder/dense_9/bias/v*Adam/autoencoder/decoder/dense_10/kernel/v(Adam/autoencoder/decoder/dense_10/bias/v*Adam/autoencoder/decoder/dense_11/kernel/v(Adam/autoencoder/decoder/dense_11/bias/v*Adam/autoencoder/decoder/dense_12/kernel/v(Adam/autoencoder/decoder/dense_12/bias/v*_
TinX
V2T*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_4606920??	
?w
?	
D__inference_encoder_layer_call_and_return_conditional_losses_4606008

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
dense/IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense/Identity?
encoder_leakyrelu_1/LeakyRelu	LeakyReludense/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2
encoder_leakyrelu_1/LeakyRelu?
encoder_leakyrelu_1/IdentityIdentity+encoder_leakyrelu_1/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
encoder_leakyrelu_1/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul%encoder_leakyrelu_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
dense_1/IdentityIdentitydense_1/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_1/Identity?
encoder_activ_layer_2/LeakyRelu	LeakyReludense_1/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_2/LeakyRelu?
encoder_activ_layer_2/IdentityIdentity-encoder_activ_layer_2/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2 
encoder_activ_layer_2/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul'encoder_activ_layer_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
dense_2/IdentityIdentitydense_2/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_2/Identity?
encoder_activ_layer_3/LeakyRelu	LeakyReludense_2/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_3/LeakyRelu?
encoder_activ_layer_3/IdentityIdentity-encoder_activ_layer_3/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2 
encoder_activ_layer_3/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul'encoder_activ_layer_3/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
dense_3/IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_3/Identity?
encoder_activ_layer_4/LeakyRelu	LeakyReludense_3/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_4/LeakyRelu?
encoder_activ_layer_4/IdentityIdentity-encoder_activ_layer_4/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2 
encoder_activ_layer_4/Identity?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul'encoder_activ_layer_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_4/BiasAdd?
dense_4/IdentityIdentitydense_4/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2
dense_4/Identity?
encoder_activ_layer_5/LeakyRelu	LeakyReludense_4/Identity:output:0*'
_output_shapes
:?????????
*
alpha%???>2!
encoder_activ_layer_5/LeakyRelu?
encoder_activ_layer_5/IdentityIdentity-encoder_activ_layer_5/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????
2 
encoder_activ_layer_5/Identity?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul'encoder_activ_layer_5/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_5/IdentityIdentitydense_5/BiasAdd:output:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_5/Identity?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul'encoder_activ_layer_5/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd?
dense_6/IdentityIdentitydense_6/BiasAdd:output:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_6/Identityi
sampling/ShapeShapedense_5/Identity:output:0*
T0*
_output_shapes
:2
sampling/Shape?
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack?
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1?
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2?
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicem
sampling/Shape_1Shapedense_5/Identity:output:0*
T0*
_output_shapes
:2
sampling/Shape_1?
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack?
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1?
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2?
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1?
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_6/Identity:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense_5/Identity:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/addv
sampling/IdentityIdentitysampling/add:z:0*
T0*'
_output_shapes
:?????????2
sampling/Identity?
IdentityIdentitydense_5/Identity:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_6/Identity:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/Identity:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:?????????::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?7
#__inference__traced_restore_4606920
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate7
3assignvariableop_5_autoencoder_encoder_dense_kernel5
1assignvariableop_6_autoencoder_encoder_dense_bias9
5assignvariableop_7_autoencoder_encoder_dense_1_kernel7
3assignvariableop_8_autoencoder_encoder_dense_1_bias9
5assignvariableop_9_autoencoder_encoder_dense_2_kernel8
4assignvariableop_10_autoencoder_encoder_dense_2_bias:
6assignvariableop_11_autoencoder_encoder_dense_3_kernel8
4assignvariableop_12_autoencoder_encoder_dense_3_bias:
6assignvariableop_13_autoencoder_encoder_dense_4_kernel8
4assignvariableop_14_autoencoder_encoder_dense_4_bias:
6assignvariableop_15_autoencoder_encoder_dense_5_kernel8
4assignvariableop_16_autoencoder_encoder_dense_5_bias:
6assignvariableop_17_autoencoder_encoder_dense_6_kernel8
4assignvariableop_18_autoencoder_encoder_dense_6_bias:
6assignvariableop_19_autoencoder_decoder_dense_7_kernel8
4assignvariableop_20_autoencoder_decoder_dense_7_bias:
6assignvariableop_21_autoencoder_decoder_dense_8_kernel8
4assignvariableop_22_autoencoder_decoder_dense_8_bias:
6assignvariableop_23_autoencoder_decoder_dense_9_kernel8
4assignvariableop_24_autoencoder_decoder_dense_9_bias;
7assignvariableop_25_autoencoder_decoder_dense_10_kernel9
5assignvariableop_26_autoencoder_decoder_dense_10_bias;
7assignvariableop_27_autoencoder_decoder_dense_11_kernel9
5assignvariableop_28_autoencoder_decoder_dense_11_bias;
7assignvariableop_29_autoencoder_decoder_dense_12_kernel9
5assignvariableop_30_autoencoder_decoder_dense_12_bias?
;assignvariableop_31_adam_autoencoder_encoder_dense_kernel_m=
9assignvariableop_32_adam_autoencoder_encoder_dense_bias_mA
=assignvariableop_33_adam_autoencoder_encoder_dense_1_kernel_m?
;assignvariableop_34_adam_autoencoder_encoder_dense_1_bias_mA
=assignvariableop_35_adam_autoencoder_encoder_dense_2_kernel_m?
;assignvariableop_36_adam_autoencoder_encoder_dense_2_bias_mA
=assignvariableop_37_adam_autoencoder_encoder_dense_3_kernel_m?
;assignvariableop_38_adam_autoencoder_encoder_dense_3_bias_mA
=assignvariableop_39_adam_autoencoder_encoder_dense_4_kernel_m?
;assignvariableop_40_adam_autoencoder_encoder_dense_4_bias_mA
=assignvariableop_41_adam_autoencoder_encoder_dense_5_kernel_m?
;assignvariableop_42_adam_autoencoder_encoder_dense_5_bias_mA
=assignvariableop_43_adam_autoencoder_encoder_dense_6_kernel_m?
;assignvariableop_44_adam_autoencoder_encoder_dense_6_bias_mA
=assignvariableop_45_adam_autoencoder_decoder_dense_7_kernel_m?
;assignvariableop_46_adam_autoencoder_decoder_dense_7_bias_mA
=assignvariableop_47_adam_autoencoder_decoder_dense_8_kernel_m?
;assignvariableop_48_adam_autoencoder_decoder_dense_8_bias_mA
=assignvariableop_49_adam_autoencoder_decoder_dense_9_kernel_m?
;assignvariableop_50_adam_autoencoder_decoder_dense_9_bias_mB
>assignvariableop_51_adam_autoencoder_decoder_dense_10_kernel_m@
<assignvariableop_52_adam_autoencoder_decoder_dense_10_bias_mB
>assignvariableop_53_adam_autoencoder_decoder_dense_11_kernel_m@
<assignvariableop_54_adam_autoencoder_decoder_dense_11_bias_mB
>assignvariableop_55_adam_autoencoder_decoder_dense_12_kernel_m@
<assignvariableop_56_adam_autoencoder_decoder_dense_12_bias_m?
;assignvariableop_57_adam_autoencoder_encoder_dense_kernel_v=
9assignvariableop_58_adam_autoencoder_encoder_dense_bias_vA
=assignvariableop_59_adam_autoencoder_encoder_dense_1_kernel_v?
;assignvariableop_60_adam_autoencoder_encoder_dense_1_bias_vA
=assignvariableop_61_adam_autoencoder_encoder_dense_2_kernel_v?
;assignvariableop_62_adam_autoencoder_encoder_dense_2_bias_vA
=assignvariableop_63_adam_autoencoder_encoder_dense_3_kernel_v?
;assignvariableop_64_adam_autoencoder_encoder_dense_3_bias_vA
=assignvariableop_65_adam_autoencoder_encoder_dense_4_kernel_v?
;assignvariableop_66_adam_autoencoder_encoder_dense_4_bias_vA
=assignvariableop_67_adam_autoencoder_encoder_dense_5_kernel_v?
;assignvariableop_68_adam_autoencoder_encoder_dense_5_bias_vA
=assignvariableop_69_adam_autoencoder_encoder_dense_6_kernel_v?
;assignvariableop_70_adam_autoencoder_encoder_dense_6_bias_vA
=assignvariableop_71_adam_autoencoder_decoder_dense_7_kernel_v?
;assignvariableop_72_adam_autoencoder_decoder_dense_7_bias_vA
=assignvariableop_73_adam_autoencoder_decoder_dense_8_kernel_v?
;assignvariableop_74_adam_autoencoder_decoder_dense_8_bias_vA
=assignvariableop_75_adam_autoencoder_decoder_dense_9_kernel_v?
;assignvariableop_76_adam_autoencoder_decoder_dense_9_bias_vB
>assignvariableop_77_adam_autoencoder_decoder_dense_10_kernel_v@
<assignvariableop_78_adam_autoencoder_decoder_dense_10_bias_vB
>assignvariableop_79_adam_autoencoder_decoder_dense_11_kernel_v@
<assignvariableop_80_adam_autoencoder_decoder_dense_11_bias_vB
>assignvariableop_81_adam_autoencoder_decoder_dense_12_kernel_v@
<assignvariableop_82_adam_autoencoder_decoder_dense_12_bias_v
identity_84??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_9?	RestoreV2?RestoreV2_1?&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?&
value?&B?%SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp3assignvariableop_5_autoencoder_encoder_dense_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_autoencoder_encoder_dense_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp5assignvariableop_7_autoencoder_encoder_dense_1_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp3assignvariableop_8_autoencoder_encoder_dense_1_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp5assignvariableop_9_autoencoder_encoder_dense_2_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp4assignvariableop_10_autoencoder_encoder_dense_2_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp6assignvariableop_11_autoencoder_encoder_dense_3_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp4assignvariableop_12_autoencoder_encoder_dense_3_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp6assignvariableop_13_autoencoder_encoder_dense_4_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp4assignvariableop_14_autoencoder_encoder_dense_4_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp6assignvariableop_15_autoencoder_encoder_dense_5_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_autoencoder_encoder_dense_5_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_autoencoder_encoder_dense_6_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_autoencoder_encoder_dense_6_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_autoencoder_decoder_dense_7_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_autoencoder_decoder_dense_7_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_autoencoder_decoder_dense_8_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_autoencoder_decoder_dense_8_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_autoencoder_decoder_dense_9_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_autoencoder_decoder_dense_9_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_autoencoder_decoder_dense_10_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp5assignvariableop_26_autoencoder_decoder_dense_10_biasIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_autoencoder_decoder_dense_11_kernelIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_autoencoder_decoder_dense_11_biasIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp7assignvariableop_29_autoencoder_decoder_dense_12_kernelIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_autoencoder_decoder_dense_12_biasIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp;assignvariableop_31_adam_autoencoder_encoder_dense_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp9assignvariableop_32_adam_autoencoder_encoder_dense_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp=assignvariableop_33_adam_autoencoder_encoder_dense_1_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp;assignvariableop_34_adam_autoencoder_encoder_dense_1_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp=assignvariableop_35_adam_autoencoder_encoder_dense_2_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp;assignvariableop_36_adam_autoencoder_encoder_dense_2_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp=assignvariableop_37_adam_autoencoder_encoder_dense_3_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp;assignvariableop_38_adam_autoencoder_encoder_dense_3_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp=assignvariableop_39_adam_autoencoder_encoder_dense_4_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp;assignvariableop_40_adam_autoencoder_encoder_dense_4_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_autoencoder_encoder_dense_5_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp;assignvariableop_42_adam_autoencoder_encoder_dense_5_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp=assignvariableop_43_adam_autoencoder_encoder_dense_6_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adam_autoencoder_encoder_dense_6_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp=assignvariableop_45_adam_autoencoder_decoder_dense_7_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp;assignvariableop_46_adam_autoencoder_decoder_dense_7_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp=assignvariableop_47_adam_autoencoder_decoder_dense_8_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp;assignvariableop_48_adam_autoencoder_decoder_dense_8_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp=assignvariableop_49_adam_autoencoder_decoder_dense_9_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp;assignvariableop_50_adam_autoencoder_decoder_dense_9_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_autoencoder_decoder_dense_10_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp<assignvariableop_52_adam_autoencoder_decoder_dense_10_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_autoencoder_decoder_dense_11_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp<assignvariableop_54_adam_autoencoder_decoder_dense_11_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_autoencoder_decoder_dense_12_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp<assignvariableop_56_adam_autoencoder_decoder_dense_12_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp;assignvariableop_57_adam_autoencoder_encoder_dense_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp9assignvariableop_58_adam_autoencoder_encoder_dense_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp=assignvariableop_59_adam_autoencoder_encoder_dense_1_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp;assignvariableop_60_adam_autoencoder_encoder_dense_1_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp=assignvariableop_61_adam_autoencoder_encoder_dense_2_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp;assignvariableop_62_adam_autoencoder_encoder_dense_2_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp=assignvariableop_63_adam_autoencoder_encoder_dense_3_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp;assignvariableop_64_adam_autoencoder_encoder_dense_3_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp=assignvariableop_65_adam_autoencoder_encoder_dense_4_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp;assignvariableop_66_adam_autoencoder_encoder_dense_4_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp=assignvariableop_67_adam_autoencoder_encoder_dense_5_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp;assignvariableop_68_adam_autoencoder_encoder_dense_5_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp=assignvariableop_69_adam_autoencoder_encoder_dense_6_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp;assignvariableop_70_adam_autoencoder_encoder_dense_6_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp=assignvariableop_71_adam_autoencoder_decoder_dense_7_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp;assignvariableop_72_adam_autoencoder_decoder_dense_7_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp=assignvariableop_73_adam_autoencoder_decoder_dense_8_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp;assignvariableop_74_adam_autoencoder_decoder_dense_8_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp=assignvariableop_75_adam_autoencoder_decoder_dense_9_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp;assignvariableop_76_adam_autoencoder_decoder_dense_9_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_autoencoder_decoder_dense_10_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp<assignvariableop_78_adam_autoencoder_decoder_dense_10_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp>assignvariableop_79_adam_autoencoder_decoder_dense_11_kernel_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp<assignvariableop_80_adam_autoencoder_decoder_dense_11_bias_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_autoencoder_decoder_dense_12_kernel_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp<assignvariableop_82_adam_autoencoder_decoder_dense_12_bias_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83?
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
%__inference_signature_wrapper_4606226
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_46059142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?d
?	
D__inference_encoder_layer_call_and_return_conditional_losses_4606300

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
encoder_leakyrelu_1/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
encoder_leakyrelu_1/LeakyRelu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul+encoder_leakyrelu_1/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd?
encoder_activ_layer_2/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_2/LeakyRelu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul-encoder_activ_layer_2/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
encoder_activ_layer_3/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_3/LeakyRelu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul-encoder_activ_layer_3/LeakyRelu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
encoder_activ_layer_4/LeakyRelu	LeakyReludense_3/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2!
encoder_activ_layer_4/LeakyRelu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul-encoder_activ_layer_4/LeakyRelu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_4/BiasAdd?
encoder_activ_layer_5/LeakyRelu	LeakyReludense_4/BiasAdd:output:0*'
_output_shapes
:?????????
*
alpha%???>2!
encoder_activ_layer_5/LeakyRelu?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul-encoder_activ_layer_5/LeakyRelu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul-encoder_activ_layer_5/LeakyRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddh
sampling/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape?
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack?
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1?
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2?
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicel
sampling/Shape_1Shapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1?
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack?
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1?
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2?
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1?
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense_5/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add?
IdentityIdentitydense_5/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_6/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/add:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:?????????::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?*
?

H__inference_autoencoder_layer_call_and_return_conditional_losses_4606157
input_1*
&encoder_statefulpartitionedcall_args_1*
&encoder_statefulpartitionedcall_args_2*
&encoder_statefulpartitionedcall_args_3*
&encoder_statefulpartitionedcall_args_4*
&encoder_statefulpartitionedcall_args_5*
&encoder_statefulpartitionedcall_args_6*
&encoder_statefulpartitionedcall_args_7*
&encoder_statefulpartitionedcall_args_8*
&encoder_statefulpartitionedcall_args_9+
'encoder_statefulpartitionedcall_args_10+
'encoder_statefulpartitionedcall_args_11+
'encoder_statefulpartitionedcall_args_12+
'encoder_statefulpartitionedcall_args_13+
'encoder_statefulpartitionedcall_args_14*
&decoder_statefulpartitionedcall_args_1*
&decoder_statefulpartitionedcall_args_2*
&decoder_statefulpartitionedcall_args_3*
&decoder_statefulpartitionedcall_args_4*
&decoder_statefulpartitionedcall_args_5*
&decoder_statefulpartitionedcall_args_6*
&decoder_statefulpartitionedcall_args_7*
&decoder_statefulpartitionedcall_args_8*
&decoder_statefulpartitionedcall_args_9+
'decoder_statefulpartitionedcall_args_10+
'decoder_statefulpartitionedcall_args_11+
'decoder_statefulpartitionedcall_args_12
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1&encoder_statefulpartitionedcall_args_1&encoder_statefulpartitionedcall_args_2&encoder_statefulpartitionedcall_args_3&encoder_statefulpartitionedcall_args_4&encoder_statefulpartitionedcall_args_5&encoder_statefulpartitionedcall_args_6&encoder_statefulpartitionedcall_args_7&encoder_statefulpartitionedcall_args_8&encoder_statefulpartitionedcall_args_9'encoder_statefulpartitionedcall_args_10'encoder_statefulpartitionedcall_args_11'encoder_statefulpartitionedcall_args_12'encoder_statefulpartitionedcall_args_13'encoder_statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????:?????????:?????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46060082!
encoder/StatefulPartitionedCall?
encoder/IdentityIdentity(encoder/StatefulPartitionedCall:output:0 ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
encoder/Identity?
encoder/Identity_1Identity(encoder/StatefulPartitionedCall:output:1 ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
encoder/Identity_1?
encoder/Identity_2Identity(encoder/StatefulPartitionedCall:output:2 ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
encoder/Identity_2?
decoder/StatefulPartitionedCallStatefulPartitionedCallencoder/Identity_2:output:0&decoder_statefulpartitionedcall_args_1&decoder_statefulpartitionedcall_args_2&decoder_statefulpartitionedcall_args_3&decoder_statefulpartitionedcall_args_4&decoder_statefulpartitionedcall_args_5&decoder_statefulpartitionedcall_args_6&decoder_statefulpartitionedcall_args_7&decoder_statefulpartitionedcall_args_8&decoder_statefulpartitionedcall_args_9'decoder_statefulpartitionedcall_args_10'decoder_statefulpartitionedcall_args_11'decoder_statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46061112!
decoder/StatefulPartitionedCall?
decoder/IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
decoder/Identityg
SquareSquareencoder/Identity:output:0*
T0*'
_output_shapes
:?????????2
Squarel
subSubencoder/Identity_1:output:0
Square:y:0*
T0*'
_output_shapes
:?????????2
sub`
ExpExpencoder/Identity_1:output:0*
T0*'
_output_shapes
:?????????2
ExpY
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
sub_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/y`
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????2
add_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xQ
mulMulmul/x:output:0Mean:output:0*
T0*
_output_shapes
: 2
mulS
div/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
div/yO
divRealDivmul:z:0div/y:output:0*
T0*
_output_shapes
: 2
div?
IdentityIdentitydecoder/Identity:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
)__inference_decoder_layer_call_fn_4606386

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_46061112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_4605914
input_1<
8autoencoder_encoder_dense_matmul_readvariableop_resource=
9autoencoder_encoder_dense_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_1_matmul_readvariableop_resource?
;autoencoder_encoder_dense_1_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_2_matmul_readvariableop_resource?
;autoencoder_encoder_dense_2_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_3_matmul_readvariableop_resource?
;autoencoder_encoder_dense_3_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_4_matmul_readvariableop_resource?
;autoencoder_encoder_dense_4_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_5_matmul_readvariableop_resource?
;autoencoder_encoder_dense_5_biasadd_readvariableop_resource>
:autoencoder_encoder_dense_6_matmul_readvariableop_resource?
;autoencoder_encoder_dense_6_biasadd_readvariableop_resource>
:autoencoder_decoder_dense_7_matmul_readvariableop_resource?
;autoencoder_decoder_dense_7_biasadd_readvariableop_resource>
:autoencoder_decoder_dense_8_matmul_readvariableop_resource?
;autoencoder_decoder_dense_8_biasadd_readvariableop_resource>
:autoencoder_decoder_dense_9_matmul_readvariableop_resource?
;autoencoder_decoder_dense_9_biasadd_readvariableop_resource?
;autoencoder_decoder_dense_10_matmul_readvariableop_resource@
<autoencoder_decoder_dense_10_biasadd_readvariableop_resource?
;autoencoder_decoder_dense_11_matmul_readvariableop_resource@
<autoencoder_decoder_dense_11_biasadd_readvariableop_resource?
;autoencoder_decoder_dense_12_matmul_readvariableop_resource@
<autoencoder_decoder_dense_12_biasadd_readvariableop_resource
identity??3autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp?2autoencoder/decoder/dense_10/MatMul/ReadVariableOp?3autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp?2autoencoder/decoder/dense_11/MatMul/ReadVariableOp?3autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp?2autoencoder/decoder/dense_12/MatMul/ReadVariableOp?2autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_7/MatMul/ReadVariableOp?2autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_8/MatMul/ReadVariableOp?2autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_9/MatMul/ReadVariableOp?0autoencoder/encoder/dense/BiasAdd/ReadVariableOp?/autoencoder/encoder/dense/MatMul/ReadVariableOp?2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_1/MatMul/ReadVariableOp?2autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_2/MatMul/ReadVariableOp?2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_3/MatMul/ReadVariableOp?2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_4/MatMul/ReadVariableOp?2autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_5/MatMul/ReadVariableOp?2autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_6/MatMul/ReadVariableOp?
/autoencoder/encoder/dense/MatMul/ReadVariableOpReadVariableOp8autoencoder_encoder_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/autoencoder/encoder/dense/MatMul/ReadVariableOp?
 autoencoder/encoder/dense/MatMulMatMulinput_17autoencoder/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 autoencoder/encoder/dense/MatMul?
0autoencoder/encoder/dense/BiasAdd/ReadVariableOpReadVariableOp9autoencoder_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0autoencoder/encoder/dense/BiasAdd/ReadVariableOp?
!autoencoder/encoder/dense/BiasAddBiasAdd*autoencoder/encoder/dense/MatMul:product:08autoencoder/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!autoencoder/encoder/dense/BiasAdd?
1autoencoder/encoder/encoder_leakyrelu_1/LeakyRelu	LeakyRelu*autoencoder/encoder/dense/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>23
1autoencoder/encoder/encoder_leakyrelu_1/LeakyRelu?
1autoencoder/encoder/dense_1/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1autoencoder/encoder/dense_1/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_1/MatMulMatMul?autoencoder/encoder/encoder_leakyrelu_1/LeakyRelu:activations:09autoencoder/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/dense_1/MatMul?
2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_1/BiasAddBiasAdd,autoencoder/encoder/dense_1/MatMul:product:0:autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/encoder/dense_1/BiasAdd?
3autoencoder/encoder/encoder_activ_layer_2/LeakyRelu	LeakyRelu,autoencoder/encoder/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>25
3autoencoder/encoder/encoder_activ_layer_2/LeakyRelu?
1autoencoder/encoder/dense_2/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1autoencoder/encoder/dense_2/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_2/MatMulMatMulAautoencoder/encoder/encoder_activ_layer_2/LeakyRelu:activations:09autoencoder/encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/dense_2/MatMul?
2autoencoder/encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_2/BiasAddBiasAdd,autoencoder/encoder/dense_2/MatMul:product:0:autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/encoder/dense_2/BiasAdd?
3autoencoder/encoder/encoder_activ_layer_3/LeakyRelu	LeakyRelu,autoencoder/encoder/dense_2/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>25
3autoencoder/encoder/encoder_activ_layer_3/LeakyRelu?
1autoencoder/encoder/dense_3/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1autoencoder/encoder/dense_3/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_3/MatMulMatMulAautoencoder/encoder/encoder_activ_layer_3/LeakyRelu:activations:09autoencoder/encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/dense_3/MatMul?
2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_3/BiasAddBiasAdd,autoencoder/encoder/dense_3/MatMul:product:0:autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/encoder/dense_3/BiasAdd?
3autoencoder/encoder/encoder_activ_layer_4/LeakyRelu	LeakyRelu,autoencoder/encoder/dense_3/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>25
3autoencoder/encoder/encoder_activ_layer_4/LeakyRelu?
1autoencoder/encoder/dense_4/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype023
1autoencoder/encoder/dense_4/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_4/MatMulMatMulAautoencoder/encoder/encoder_activ_layer_4/LeakyRelu:activations:09autoencoder/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2$
"autoencoder/encoder/dense_4/MatMul?
2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype024
2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_4/BiasAddBiasAdd,autoencoder/encoder/dense_4/MatMul:product:0:autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2%
#autoencoder/encoder/dense_4/BiasAdd?
3autoencoder/encoder/encoder_activ_layer_5/LeakyRelu	LeakyRelu,autoencoder/encoder/dense_4/BiasAdd:output:0*'
_output_shapes
:?????????
*
alpha%???>25
3autoencoder/encoder/encoder_activ_layer_5/LeakyRelu?
1autoencoder/encoder/dense_5/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype023
1autoencoder/encoder/dense_5/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_5/MatMulMatMulAautoencoder/encoder/encoder_activ_layer_5/LeakyRelu:activations:09autoencoder/encoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/dense_5/MatMul?
2autoencoder/encoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_5/BiasAddBiasAdd,autoencoder/encoder/dense_5/MatMul:product:0:autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/encoder/dense_5/BiasAdd?
1autoencoder/encoder/dense_6/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype023
1autoencoder/encoder/dense_6/MatMul/ReadVariableOp?
"autoencoder/encoder/dense_6/MatMulMatMulAautoencoder/encoder/encoder_activ_layer_5/LeakyRelu:activations:09autoencoder/encoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/dense_6/MatMul?
2autoencoder/encoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp?
#autoencoder/encoder/dense_6/BiasAddBiasAdd,autoencoder/encoder/dense_6/MatMul:product:0:autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/encoder/dense_6/BiasAdd?
"autoencoder/encoder/sampling/ShapeShape,autoencoder/encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2$
"autoencoder/encoder/sampling/Shape?
0autoencoder/encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0autoencoder/encoder/sampling/strided_slice/stack?
2autoencoder/encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2autoencoder/encoder/sampling/strided_slice/stack_1?
2autoencoder/encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2autoencoder/encoder/sampling/strided_slice/stack_2?
*autoencoder/encoder/sampling/strided_sliceStridedSlice+autoencoder/encoder/sampling/Shape:output:09autoencoder/encoder/sampling/strided_slice/stack:output:0;autoencoder/encoder/sampling/strided_slice/stack_1:output:0;autoencoder/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*autoencoder/encoder/sampling/strided_slice?
$autoencoder/encoder/sampling/Shape_1Shape,autoencoder/encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2&
$autoencoder/encoder/sampling/Shape_1?
2autoencoder/encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2autoencoder/encoder/sampling/strided_slice_1/stack?
4autoencoder/encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4autoencoder/encoder/sampling/strided_slice_1/stack_1?
4autoencoder/encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4autoencoder/encoder/sampling/strided_slice_1/stack_2?
,autoencoder/encoder/sampling/strided_slice_1StridedSlice-autoencoder/encoder/sampling/Shape_1:output:0;autoencoder/encoder/sampling/strided_slice_1/stack:output:0=autoencoder/encoder/sampling/strided_slice_1/stack_1:output:0=autoencoder/encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,autoencoder/encoder/sampling/strided_slice_1?
0autoencoder/encoder/sampling/random_normal/shapePack3autoencoder/encoder/sampling/strided_slice:output:05autoencoder/encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:22
0autoencoder/encoder/sampling/random_normal/shape?
/autoencoder/encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/autoencoder/encoder/sampling/random_normal/mean?
1autoencoder/encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1autoencoder/encoder/sampling/random_normal/stddev?
?autoencoder/encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal9autoencoder/encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed???)*
seed2???2A
?autoencoder/encoder/sampling/random_normal/RandomStandardNormal?
.autoencoder/encoder/sampling/random_normal/mulMulHautoencoder/encoder/sampling/random_normal/RandomStandardNormal:output:0:autoencoder/encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????20
.autoencoder/encoder/sampling/random_normal/mul?
*autoencoder/encoder/sampling/random_normalAdd2autoencoder/encoder/sampling/random_normal/mul:z:08autoencoder/encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2,
*autoencoder/encoder/sampling/random_normal?
"autoencoder/encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"autoencoder/encoder/sampling/mul/x?
 autoencoder/encoder/sampling/mulMul+autoencoder/encoder/sampling/mul/x:output:0,autoencoder/encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 autoencoder/encoder/sampling/mul?
 autoencoder/encoder/sampling/ExpExp$autoencoder/encoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2"
 autoencoder/encoder/sampling/Exp?
"autoencoder/encoder/sampling/mul_1Mul$autoencoder/encoder/sampling/Exp:y:0.autoencoder/encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/encoder/sampling/mul_1?
 autoencoder/encoder/sampling/addAddV2,autoencoder/encoder/dense_5/BiasAdd:output:0&autoencoder/encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2"
 autoencoder/encoder/sampling/add?
1autoencoder/decoder/dense_7/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype023
1autoencoder/decoder/dense_7/MatMul/ReadVariableOp?
"autoencoder/decoder/dense_7/MatMulMatMul$autoencoder/encoder/sampling/add:z:09autoencoder/decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2$
"autoencoder/decoder/dense_7/MatMul?
2autoencoder/decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype024
2autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp?
#autoencoder/decoder/dense_7/BiasAddBiasAdd,autoencoder/decoder/dense_7/MatMul:product:0:autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2%
#autoencoder/decoder/dense_7/BiasAdd?
1autoencoder/decoder/decoder_leakyrelu_1/LeakyRelu	LeakyRelu,autoencoder/decoder/dense_7/BiasAdd:output:0*'
_output_shapes
:?????????
*
alpha%???>23
1autoencoder/decoder/decoder_leakyrelu_1/LeakyRelu?
1autoencoder/decoder/dense_8/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_8_matmul_readvariableop_resource*
_output_shapes

:
*
dtype023
1autoencoder/decoder/dense_8/MatMul/ReadVariableOp?
"autoencoder/decoder/dense_8/MatMulMatMul?autoencoder/decoder/decoder_leakyrelu_1/LeakyRelu:activations:09autoencoder/decoder/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/decoder/dense_8/MatMul?
2autoencoder/decoder/dense_8/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp?
#autoencoder/decoder/dense_8/BiasAddBiasAdd,autoencoder/decoder/dense_8/MatMul:product:0:autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/decoder/dense_8/BiasAdd?
1autoencoder/decoder/decoder_leakyrelu_2/LeakyRelu	LeakyRelu,autoencoder/decoder/dense_8/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>23
1autoencoder/decoder/decoder_leakyrelu_2/LeakyRelu?
1autoencoder/decoder/dense_9/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1autoencoder/decoder/dense_9/MatMul/ReadVariableOp?
"autoencoder/decoder/dense_9/MatMulMatMul?autoencoder/decoder/decoder_leakyrelu_2/LeakyRelu:activations:09autoencoder/decoder/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"autoencoder/decoder/dense_9/MatMul?
2autoencoder/decoder/dense_9/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp?
#autoencoder/decoder/dense_9/BiasAddBiasAdd,autoencoder/decoder/dense_9/MatMul:product:0:autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/decoder/dense_9/BiasAdd?
1autoencoder/decoder/decoder_leakyrelu_3/LeakyRelu	LeakyRelu,autoencoder/decoder/dense_9/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>23
1autoencoder/decoder/decoder_leakyrelu_3/LeakyRelu?
2autoencoder/decoder/dense_10/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2autoencoder/decoder/dense_10/MatMul/ReadVariableOp?
#autoencoder/decoder/dense_10/MatMulMatMul?autoencoder/decoder/decoder_leakyrelu_3/LeakyRelu:activations:0:autoencoder/decoder/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/decoder/dense_10/MatMul?
3autoencoder/decoder/dense_10/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp?
$autoencoder/decoder/dense_10/BiasAddBiasAdd-autoencoder/decoder/dense_10/MatMul:product:0;autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$autoencoder/decoder/dense_10/BiasAdd?
1autoencoder/decoder/decoder_leakyrelu_4/LeakyRelu	LeakyRelu-autoencoder/decoder/dense_10/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>23
1autoencoder/decoder/decoder_leakyrelu_4/LeakyRelu?
2autoencoder/decoder/dense_11/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2autoencoder/decoder/dense_11/MatMul/ReadVariableOp?
#autoencoder/decoder/dense_11/MatMulMatMul?autoencoder/decoder/decoder_leakyrelu_4/LeakyRelu:activations:0:autoencoder/decoder/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/decoder/dense_11/MatMul?
3autoencoder/decoder/dense_11/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp?
$autoencoder/decoder/dense_11/BiasAddBiasAdd-autoencoder/decoder/dense_11/MatMul:product:0;autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$autoencoder/decoder/dense_11/BiasAdd?
1autoencoder/decoder/decoder_leakyrelu_5/LeakyRelu	LeakyRelu-autoencoder/decoder/dense_11/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>23
1autoencoder/decoder/decoder_leakyrelu_5/LeakyRelu?
2autoencoder/decoder/dense_12/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2autoencoder/decoder/dense_12/MatMul/ReadVariableOp?
#autoencoder/decoder/dense_12/MatMulMatMul?autoencoder/decoder/decoder_leakyrelu_5/LeakyRelu:activations:0:autoencoder/decoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#autoencoder/decoder/dense_12/MatMul?
3autoencoder/decoder/dense_12/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp?
$autoencoder/decoder/dense_12/BiasAddBiasAdd-autoencoder/decoder/dense_12/MatMul:product:0;autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$autoencoder/decoder/dense_12/BiasAdd?
6autoencoder/decoder/decoder_leakyrelu_output/LeakyRelu	LeakyRelu-autoencoder/decoder/dense_12/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>28
6autoencoder/decoder/decoder_leakyrelu_output/LeakyRelu?
autoencoder/SquareSquare,autoencoder/encoder/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
autoencoder/Square?
autoencoder/subSub,autoencoder/encoder/dense_6/BiasAdd:output:0autoencoder/Square:y:0*
T0*'
_output_shapes
:?????????2
autoencoder/sub?
autoencoder/ExpExp,autoencoder/encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
autoencoder/Exp?
autoencoder/sub_1Subautoencoder/sub:z:0autoencoder/Exp:y:0*
T0*'
_output_shapes
:?????????2
autoencoder/sub_1k
autoencoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
autoencoder/add/y?
autoencoder/addAddV2autoencoder/sub_1:z:0autoencoder/add/y:output:0*
T0*'
_output_shapes
:?????????2
autoencoder/addw
autoencoder/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
autoencoder/Const~
autoencoder/MeanMeanautoencoder/add:z:0autoencoder/Const:output:0*
T0*
_output_shapes
: 2
autoencoder/Meank
autoencoder/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
autoencoder/mul/x?
autoencoder/mulMulautoencoder/mul/x:output:0autoencoder/Mean:output:0*
T0*
_output_shapes
: 2
autoencoder/mulk
autoencoder/div/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2
autoencoder/div/y
autoencoder/divRealDivautoencoder/mul:z:0autoencoder/div/y:output:0*
T0*
_output_shapes
: 2
autoencoder/div?
IdentityIdentityDautoencoder/decoder/decoder_leakyrelu_output/LeakyRelu:activations:04^autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_10/MatMul/ReadVariableOp4^autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_11/MatMul/ReadVariableOp4^autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_12/MatMul/ReadVariableOp3^autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_7/MatMul/ReadVariableOp3^autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_8/MatMul/ReadVariableOp3^autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_9/MatMul/ReadVariableOp1^autoencoder/encoder/dense/BiasAdd/ReadVariableOp0^autoencoder/encoder/dense/MatMul/ReadVariableOp3^autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_1/MatMul/ReadVariableOp3^autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_2/MatMul/ReadVariableOp3^autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_3/MatMul/ReadVariableOp3^autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_4/MatMul/ReadVariableOp3^autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_5/MatMul/ReadVariableOp3^autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::2j
3autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_10/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_10/MatMul/ReadVariableOp2autoencoder/decoder/dense_10/MatMul/ReadVariableOp2j
3autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_11/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_11/MatMul/ReadVariableOp2autoencoder/decoder/dense_11/MatMul/ReadVariableOp2j
3autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_12/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_12/MatMul/ReadVariableOp2autoencoder/decoder/dense_12/MatMul/ReadVariableOp2h
2autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_7/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_7/MatMul/ReadVariableOp1autoencoder/decoder/dense_7/MatMul/ReadVariableOp2h
2autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_8/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_8/MatMul/ReadVariableOp1autoencoder/decoder/dense_8/MatMul/ReadVariableOp2h
2autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_9/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_9/MatMul/ReadVariableOp1autoencoder/decoder/dense_9/MatMul/ReadVariableOp2d
0autoencoder/encoder/dense/BiasAdd/ReadVariableOp0autoencoder/encoder/dense/BiasAdd/ReadVariableOp2b
/autoencoder/encoder/dense/MatMul/ReadVariableOp/autoencoder/encoder/dense/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_1/MatMul/ReadVariableOp1autoencoder/encoder/dense_1/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_2/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_2/MatMul/ReadVariableOp1autoencoder/encoder/dense_2/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_3/MatMul/ReadVariableOp1autoencoder/encoder/dense_3/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_4/MatMul/ReadVariableOp1autoencoder/encoder/dense_4/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_5/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_5/MatMul/ReadVariableOp1autoencoder/encoder/dense_5/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_6/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_6/MatMul/ReadVariableOp1autoencoder/encoder/dense_6/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
?
?
)__inference_encoder_layer_call_fn_4606323

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????:?????????:?????????**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_46060082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_4606186
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_46061572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes}
{:?????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?L
?
D__inference_decoder_layer_call_and_return_conditional_losses_4606111

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/BiasAdd?
dense_7/IdentityIdentitydense_7/BiasAdd:output:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2
dense_7/Identity?
decoder_leakyrelu_1/LeakyRelu	LeakyReludense_7/Identity:output:0*'
_output_shapes
:?????????
*
alpha%???>2
decoder_leakyrelu_1/LeakyRelu?
decoder_leakyrelu_1/IdentityIdentity+decoder_leakyrelu_1/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????
2
decoder_leakyrelu_1/Identity?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul%decoder_leakyrelu_1/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd?
dense_8/IdentityIdentitydense_8/BiasAdd:output:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_8/Identity?
decoder_leakyrelu_2/LeakyRelu	LeakyReludense_8/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_2/LeakyRelu?
decoder_leakyrelu_2/IdentityIdentity+decoder_leakyrelu_2/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
decoder_leakyrelu_2/Identity?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul%decoder_leakyrelu_2/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd?
dense_9/IdentityIdentitydense_9/BiasAdd:output:0^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_9/Identity?
decoder_leakyrelu_3/LeakyRelu	LeakyReludense_9/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_3/LeakyRelu?
decoder_leakyrelu_3/IdentityIdentity+decoder_leakyrelu_3/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
decoder_leakyrelu_3/Identity?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul%decoder_leakyrelu_3/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdd?
dense_10/IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_10/Identity?
decoder_leakyrelu_4/LeakyRelu	LeakyReludense_10/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_4/LeakyRelu?
decoder_leakyrelu_4/IdentityIdentity+decoder_leakyrelu_4/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
decoder_leakyrelu_4/Identity?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul%decoder_leakyrelu_4/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd?
dense_11/IdentityIdentitydense_11/BiasAdd:output:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_11/Identity?
decoder_leakyrelu_5/LeakyRelu	LeakyReludense_11/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_5/LeakyRelu?
decoder_leakyrelu_5/IdentityIdentity+decoder_leakyrelu_5/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
decoder_leakyrelu_5/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul%decoder_leakyrelu_5/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
dense_12/IdentityIdentitydense_12/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
dense_12/Identity?
"decoder_leakyrelu_output/LeakyRelu	LeakyReludense_12/Identity:output:0*'
_output_shapes
:?????????*
alpha%???>2$
"decoder_leakyrelu_output/LeakyRelu?
!decoder_leakyrelu_output/IdentityIdentity0decoder_leakyrelu_output/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2#
!decoder_leakyrelu_output/Identity?
IdentityIdentity*decoder_leakyrelu_output/Identity:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?-
 __inference__traced_save_4606659
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop?
;savev2_autoencoder_encoder_dense_kernel_read_readvariableop=
9savev2_autoencoder_encoder_dense_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_1_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_1_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_2_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_2_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_3_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_3_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_4_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_4_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_5_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_5_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_6_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_6_bias_read_readvariableopA
=savev2_autoencoder_decoder_dense_7_kernel_read_readvariableop?
;savev2_autoencoder_decoder_dense_7_bias_read_readvariableopA
=savev2_autoencoder_decoder_dense_8_kernel_read_readvariableop?
;savev2_autoencoder_decoder_dense_8_bias_read_readvariableopA
=savev2_autoencoder_decoder_dense_9_kernel_read_readvariableop?
;savev2_autoencoder_decoder_dense_9_bias_read_readvariableopB
>savev2_autoencoder_decoder_dense_10_kernel_read_readvariableop@
<savev2_autoencoder_decoder_dense_10_bias_read_readvariableopB
>savev2_autoencoder_decoder_dense_11_kernel_read_readvariableop@
<savev2_autoencoder_decoder_dense_11_bias_read_readvariableopB
>savev2_autoencoder_decoder_dense_12_kernel_read_readvariableop@
<savev2_autoencoder_decoder_dense_12_bias_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_kernel_m_read_readvariableopD
@savev2_adam_autoencoder_encoder_dense_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_1_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_1_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_2_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_2_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_3_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_3_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_4_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_4_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_5_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_5_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_6_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_6_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_7_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_7_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_8_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_8_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_9_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_9_bias_m_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_10_kernel_m_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_10_bias_m_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_11_kernel_m_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_11_bias_m_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_12_kernel_m_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_12_bias_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_kernel_v_read_readvariableopD
@savev2_adam_autoencoder_encoder_dense_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_1_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_1_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_2_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_2_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_3_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_3_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_4_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_4_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_5_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_5_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_6_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_6_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_7_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_7_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_8_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_8_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_9_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_9_bias_v_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_10_kernel_v_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_10_bias_v_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_11_kernel_v_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_11_bias_v_read_readvariableopI
Esavev2_adam_autoencoder_decoder_dense_12_kernel_v_read_readvariableopG
Csavev2_adam_autoencoder_decoder_dense_12_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_952461d915d34c5c8cde9a9d32b0407e/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?&
value?&B?%SB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop;savev2_autoencoder_encoder_dense_kernel_read_readvariableop9savev2_autoencoder_encoder_dense_bias_read_readvariableop=savev2_autoencoder_encoder_dense_1_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_1_bias_read_readvariableop=savev2_autoencoder_encoder_dense_2_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_2_bias_read_readvariableop=savev2_autoencoder_encoder_dense_3_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_3_bias_read_readvariableop=savev2_autoencoder_encoder_dense_4_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_4_bias_read_readvariableop=savev2_autoencoder_encoder_dense_5_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_5_bias_read_readvariableop=savev2_autoencoder_encoder_dense_6_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_6_bias_read_readvariableop=savev2_autoencoder_decoder_dense_7_kernel_read_readvariableop;savev2_autoencoder_decoder_dense_7_bias_read_readvariableop=savev2_autoencoder_decoder_dense_8_kernel_read_readvariableop;savev2_autoencoder_decoder_dense_8_bias_read_readvariableop=savev2_autoencoder_decoder_dense_9_kernel_read_readvariableop;savev2_autoencoder_decoder_dense_9_bias_read_readvariableop>savev2_autoencoder_decoder_dense_10_kernel_read_readvariableop<savev2_autoencoder_decoder_dense_10_bias_read_readvariableop>savev2_autoencoder_decoder_dense_11_kernel_read_readvariableop<savev2_autoencoder_decoder_dense_11_bias_read_readvariableop>savev2_autoencoder_decoder_dense_12_kernel_read_readvariableop<savev2_autoencoder_decoder_dense_12_bias_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_kernel_m_read_readvariableop@savev2_adam_autoencoder_encoder_dense_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_1_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_1_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_2_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_2_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_3_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_3_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_4_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_4_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_5_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_5_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_6_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_6_bias_m_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_7_kernel_m_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_7_bias_m_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_8_kernel_m_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_8_bias_m_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_9_kernel_m_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_9_bias_m_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_10_kernel_m_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_10_bias_m_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_11_kernel_m_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_11_bias_m_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_12_kernel_m_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_12_bias_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_kernel_v_read_readvariableop@savev2_adam_autoencoder_encoder_dense_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_1_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_1_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_2_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_2_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_3_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_3_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_4_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_4_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_5_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_5_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_6_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_6_bias_v_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_7_kernel_v_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_7_bias_v_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_8_kernel_v_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_8_bias_v_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_9_kernel_v_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_9_bias_v_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_10_kernel_v_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_10_bias_v_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_11_kernel_v_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_11_bias_v_read_readvariableopEsavev2_adam_autoencoder_decoder_dense_12_kernel_v_read_readvariableopCsavev2_adam_autoencoder_decoder_dense_12_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :::::::::
:
:
::
::
:
:
::::::::::::::::::
:
:
::
::
:
:
::::::::::::::::::
:
:
::
::
:
:
:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?;
?
D__inference_decoder_layer_call_and_return_conditional_losses_4606369

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_7/BiasAdd?
decoder_leakyrelu_1/LeakyRelu	LeakyReludense_7/BiasAdd:output:0*'
_output_shapes
:?????????
*
alpha%???>2
decoder_leakyrelu_1/LeakyRelu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul+decoder_leakyrelu_1/LeakyRelu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd?
decoder_leakyrelu_2/LeakyRelu	LeakyReludense_8/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_2/LeakyRelu?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul+decoder_leakyrelu_2/LeakyRelu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAdd?
decoder_leakyrelu_3/LeakyRelu	LeakyReludense_9/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_3/LeakyRelu?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul+decoder_leakyrelu_3/LeakyRelu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/BiasAdd?
decoder_leakyrelu_4/LeakyRelu	LeakyReludense_10/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_4/LeakyRelu?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul+decoder_leakyrelu_4/LeakyRelu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/BiasAdd?
decoder_leakyrelu_5/LeakyRelu	LeakyReludense_11/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
decoder_leakyrelu_5/LeakyRelu?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul+decoder_leakyrelu_5/LeakyRelu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
"decoder_leakyrelu_output/LeakyRelu	LeakyReludense_12/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2$
"decoder_leakyrelu_output/LeakyRelu?
IdentityIdentity0decoder_leakyrelu_output/LeakyRelu:activations:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ޖ
?
encoder
decoder
	optimizer

signatures
	variables
regularization_losses
	keras_api
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_model?{"training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "loss_weights": null, "metrics": [], "optimizer_config": {"class_name": "Adam", "config": {"beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "name": "Adam", "decay": 0.0, "epsilon": 1e-07, "learning_rate": 0.0005000000237487257, "amsgrad": false}}, "weighted_metrics": null, "sample_weight_mode": null}, "name": "autoencoder", "class_name": "VariationalAutoEncoder", "dtype": "float32", "is_graph_network": false, "trainable": true, "model_config": {"class_name": "VariationalAutoEncoder"}, "expects_training_arg": false, "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow"}
?
	encoder_layer1

encoder_active_layer1
encoder_layer2
encoder_active_layer2
encoder_layer3
encoder_active_layer3
encoder_layer4
encoder_active_layer4
encoder_layer5
encoder_active_layer5

dense_mean
dense_log_var
sampling
	variables
regularization_losses
	keras_api
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder", "class_name": "Encoder", "dtype": "float32", "batch_input_shape": null, "trainable": true, "expects_training_arg": false}
?
decoder_layer1
decoder_active_layer1
decoder_layer2
decoder_active_layer2
decoder_layer3
decoder_active_layer3
 decoder_layer4
!decoder_active_layer4
"decoder_layer5
#decoder_active_layer5
$dense_output
%decoder_active_output
&	variables
'regularization_losses
(	keras_api
)trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder", "class_name": "Decoder", "dtype": "float32", "batch_input_shape": null, "trainable": true, "expects_training_arg": false}
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?"
	optimizer
-
?serving_default"
signature_map
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ilayers
trainable_variables
	variables
Jlayer_regularization_losses
Kmetrics
regularization_losses
Lnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
=14
>15
?16
@17
A18
B19
C20
D21
E22
F23
G24
H25"
trackable_list_wrapper
?

/kernel
0bias
M	variables
Nregularization_losses
O	keras_api
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 30, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 14}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
Q	variables
Rregularization_losses
S	keras_api
Ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder_leakyrelu_1", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "encoder_leakyrelu_1", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

1kernel
2bias
U	variables
Vregularization_losses
W	keras_api
Xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_1", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 30}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
Y	variables
Zregularization_losses
[	keras_api
\trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder_activ_layer_2", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "encoder_activ_layer_2", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

3kernel
4bias
]	variables
^regularization_losses
_	keras_api
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_2", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
a	variables
bregularization_losses
c	keras_api
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder_activ_layer_3", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "encoder_activ_layer_3", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

5kernel
6bias
e	variables
fregularization_losses
g	keras_api
htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_3", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
i	variables
jregularization_losses
k	keras_api
ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder_activ_layer_4", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "encoder_activ_layer_4", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

7kernel
8bias
m	variables
nregularization_losses
o	keras_api
ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_4", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
q	variables
rregularization_losses
s	keras_api
ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder_activ_layer_5", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "encoder_activ_layer_5", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

9kernel
:bias
u	variables
vregularization_losses
w	keras_api
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_5", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 5, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 10}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?

;kernel
<bias
y	variables
zregularization_losses
{	keras_api
|trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_6", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_6", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 5, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 10}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
}	variables
~regularization_losses
	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "sampling", "class_name": "sampling", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "sampling"}, "expects_training_arg": false}
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
	variables
 ?layer_regularization_losses
?metrics
regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13"
trackable_list_wrapper
?

=kernel
>bias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_7", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_7", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 5}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_1", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_1", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

?kernel
@bias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_8", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_8", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 10}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_2", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_2", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

Akernel
Bbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_9", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_9", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_3", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_3", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

Ckernel
Dbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_10", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_10", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_4", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_4", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

Ekernel
Fbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_11", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_11", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 30, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 20}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_5", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_5", "alpha": 0.30000001192092896}, "expects_training_arg": false}
?

Gkernel
Hbias
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_12", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_12", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 14, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 30}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
?
?	variables
?regularization_losses
?	keras_api
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder_leakyrelu_output", "class_name": "LeakyReLU", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"dtype": "float32", "trainable": true, "name": "decoder_leakyrelu_output", "alpha": 0.30000001192092896}, "expects_training_arg": false}
v
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
)trainable_variables
&	variables
 ?layer_regularization_losses
?metrics
'regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
v
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11"
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
2:02 autoencoder/encoder/dense/kernel
,:*2autoencoder/encoder/dense/bias
4:22"autoencoder/encoder/dense_1/kernel
.:,2 autoencoder/encoder/dense_1/bias
4:22"autoencoder/encoder/dense_2/kernel
.:,2 autoencoder/encoder/dense_2/bias
4:22"autoencoder/encoder/dense_3/kernel
.:,2 autoencoder/encoder/dense_3/bias
4:2
2"autoencoder/encoder/dense_4/kernel
.:,
2 autoencoder/encoder/dense_4/bias
4:2
2"autoencoder/encoder/dense_5/kernel
.:,2 autoencoder/encoder/dense_5/bias
4:2
2"autoencoder/encoder/dense_6/kernel
.:,2 autoencoder/encoder/dense_6/bias
4:2
2"autoencoder/decoder/dense_7/kernel
.:,
2 autoencoder/decoder/dense_7/bias
4:2
2"autoencoder/decoder/dense_8/kernel
.:,2 autoencoder/decoder/dense_8/bias
4:22"autoencoder/decoder/dense_9/kernel
.:,2 autoencoder/decoder/dense_9/bias
5:32#autoencoder/decoder/dense_10/kernel
/:-2!autoencoder/decoder/dense_10/bias
5:32#autoencoder/decoder/dense_11/kernel
/:-2!autoencoder/decoder/dense_11/bias
5:32#autoencoder/decoder/dense_12/kernel
/:-2!autoencoder/decoder/dense_12/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Ptrainable_variables
M	variables
 ?layer_regularization_losses
?metrics
Nregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Ttrainable_variables
Q	variables
 ?layer_regularization_losses
?metrics
Rregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Xtrainable_variables
U	variables
 ?layer_regularization_losses
?metrics
Vregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
\trainable_variables
Y	variables
 ?layer_regularization_losses
?metrics
Zregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
`trainable_variables
]	variables
 ?layer_regularization_losses
?metrics
^regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
dtrainable_variables
a	variables
 ?layer_regularization_losses
?metrics
bregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
htrainable_variables
e	variables
 ?layer_regularization_losses
?metrics
fregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
ltrainable_variables
i	variables
 ?layer_regularization_losses
?metrics
jregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
ptrainable_variables
m	variables
 ?layer_regularization_losses
?metrics
nregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
ttrainable_variables
q	variables
 ?layer_regularization_losses
?metrics
rregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
xtrainable_variables
u	variables
 ?layer_regularization_losses
?metrics
vregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
|trainable_variables
y	variables
 ?layer_regularization_losses
?metrics
zregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
}	variables
 ?layer_regularization_losses
?metrics
~regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?metrics
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
v
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7:52'Adam/autoencoder/encoder/dense/kernel/m
1:/2%Adam/autoencoder/encoder/dense/bias/m
9:72)Adam/autoencoder/encoder/dense_1/kernel/m
3:12'Adam/autoencoder/encoder/dense_1/bias/m
9:72)Adam/autoencoder/encoder/dense_2/kernel/m
3:12'Adam/autoencoder/encoder/dense_2/bias/m
9:72)Adam/autoencoder/encoder/dense_3/kernel/m
3:12'Adam/autoencoder/encoder/dense_3/bias/m
9:7
2)Adam/autoencoder/encoder/dense_4/kernel/m
3:1
2'Adam/autoencoder/encoder/dense_4/bias/m
9:7
2)Adam/autoencoder/encoder/dense_5/kernel/m
3:12'Adam/autoencoder/encoder/dense_5/bias/m
9:7
2)Adam/autoencoder/encoder/dense_6/kernel/m
3:12'Adam/autoencoder/encoder/dense_6/bias/m
9:7
2)Adam/autoencoder/decoder/dense_7/kernel/m
3:1
2'Adam/autoencoder/decoder/dense_7/bias/m
9:7
2)Adam/autoencoder/decoder/dense_8/kernel/m
3:12'Adam/autoencoder/decoder/dense_8/bias/m
9:72)Adam/autoencoder/decoder/dense_9/kernel/m
3:12'Adam/autoencoder/decoder/dense_9/bias/m
::82*Adam/autoencoder/decoder/dense_10/kernel/m
4:22(Adam/autoencoder/decoder/dense_10/bias/m
::82*Adam/autoencoder/decoder/dense_11/kernel/m
4:22(Adam/autoencoder/decoder/dense_11/bias/m
::82*Adam/autoencoder/decoder/dense_12/kernel/m
4:22(Adam/autoencoder/decoder/dense_12/bias/m
7:52'Adam/autoencoder/encoder/dense/kernel/v
1:/2%Adam/autoencoder/encoder/dense/bias/v
9:72)Adam/autoencoder/encoder/dense_1/kernel/v
3:12'Adam/autoencoder/encoder/dense_1/bias/v
9:72)Adam/autoencoder/encoder/dense_2/kernel/v
3:12'Adam/autoencoder/encoder/dense_2/bias/v
9:72)Adam/autoencoder/encoder/dense_3/kernel/v
3:12'Adam/autoencoder/encoder/dense_3/bias/v
9:7
2)Adam/autoencoder/encoder/dense_4/kernel/v
3:1
2'Adam/autoencoder/encoder/dense_4/bias/v
9:7
2)Adam/autoencoder/encoder/dense_5/kernel/v
3:12'Adam/autoencoder/encoder/dense_5/bias/v
9:7
2)Adam/autoencoder/encoder/dense_6/kernel/v
3:12'Adam/autoencoder/encoder/dense_6/bias/v
9:7
2)Adam/autoencoder/decoder/dense_7/kernel/v
3:1
2'Adam/autoencoder/decoder/dense_7/bias/v
9:7
2)Adam/autoencoder/decoder/dense_8/kernel/v
3:12'Adam/autoencoder/decoder/dense_8/bias/v
9:72)Adam/autoencoder/decoder/dense_9/kernel/v
3:12'Adam/autoencoder/decoder/dense_9/bias/v
::82*Adam/autoencoder/decoder/dense_10/kernel/v
4:22(Adam/autoencoder/decoder/dense_10/bias/v
::82*Adam/autoencoder/decoder/dense_11/kernel/v
4:22(Adam/autoencoder/decoder/dense_11/bias/v
::82*Adam/autoencoder/decoder/dense_12/kernel/v
4:22(Adam/autoencoder/decoder/dense_12/bias/v
?2?
-__inference_autoencoder_layer_call_fn_4606186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
H__inference_autoencoder_layer_call_and_return_conditional_losses_4606157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
"__inference__wrapped_model_4605914?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
)__inference_encoder_layer_call_fn_4606323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_4606300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_decoder_layer_call_fn_4606386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_4606369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
4B2
%__inference_signature_wrapper_4606226input_1
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4605914?/0123456789:;<=>?@ABCDEFGH0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
H__inference_autoencoder_layer_call_and_return_conditional_losses_4606157u/0123456789:;<=>?@ABCDEFGH0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
-__inference_autoencoder_layer_call_fn_4606186h/0123456789:;<=>?@ABCDEFGH0?-
&?#
!?
input_1?????????
? "???????????
D__inference_decoder_layer_call_and_return_conditional_losses_4606369f=>?@ABCDEFGH/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
)__inference_decoder_layer_call_fn_4606386Y=>?@ABCDEFGH/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_encoder_layer_call_and_return_conditional_losses_4606300?/0123456789:;</?,
%?"
 ?
inputs?????????
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
)__inference_encoder_layer_call_fn_4606323?/0123456789:;</?,
%?"
 ?
inputs?????????
? "Z?W
?
0?????????
?
1?????????
?
2??????????
%__inference_signature_wrapper_4606226?/0123456789:;<=>?@ABCDEFGH;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????