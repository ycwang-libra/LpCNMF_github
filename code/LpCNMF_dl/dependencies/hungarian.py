# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:04:18 2021

@author: yc_wang
"""
import numpy as np
from dependencies.utils import LogiMul1d, Concat_Array322, MATLAB_sparse,\
	MATLAB_hungarian187, LogiMul2d, Mat_hungarian415, Mat_hungarian418

def hungarian(A):
# %HUNGARIAN Solve the Assignment problem using the Hungarian method.
# %
# %[C,T]=hungarian(A)
# %A - a square cost matrix.
# %C - the optimal assignment.
# %T - the cost of the optimal assignment.
# %s.t. T = trace(A(C,:)) is minimized over all possible assignments.

# % Adapted from the FORTRAN IV code in Carpaneto and Toth, "Algorithm 548:
# % Solution of the assignment problem [H]", ACM Transactions on
# % Mathematical Software, 6(1):104-111, 1980.

# % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
# %                 Department of Computing Science, Umeå University,
# %                 Sweden. 
# %                 All standard disclaimers apply.

# % A substantial effort was put into this code. If you use it for a
# % publication or otherwise, please include an acknowledgement or at least
# % notify me by email. /Niclas
# % Translated by Yicheng Wang on 26th Oct 2021
	m, n = A.shape
	if m != n:
		print('HUNGARIAN: Cost matrix must be square!')

	# Save original cost matrix.
	orig = A.copy()

	# Reduce matrix.
	A = hminired(A)  # debugged

	# Do an initial assignment.
	A, C, U = hminiass(A)  # debugged

	# Repeat while we have unassigned rows.
	
	while U[n]:
		
		LR = np.zeros([n])
		LC = np.zeros([n])
		CH = np.zeros([n])
		RH = np.hstack((np.zeros([n]),np.array([-1])))

		# No labelled columns.
		SLC = []
		
		# Start path in first unassigned row.
		r = int(U[n])
		# Mark row with end-of-path label.
		LR[r-1] = -1                         ##??? DEBUG LR[r] = -1
		# Insert row first in labelled row set.
		SLR = [r] # 

		# Repeat until we manage to find an assignable zero.
		while 1:
			# If there are free zeros in row r
			if A[r-1,n] != 0:
				#  ...get column of first free zero.
				l = int(-A[r-1,n]) 

				# If there are more free zeros in row r and row r in not
				# yet marked as unexplored..
				if A[r-1,l-1] != 0 and RH[r-1] == 0:
					# Insert row r first in unexplored list.
					RH[r-1] = RH[n]
					RH[n] = r

					# Mark in which column the next unexplored zero in this row is.
					CH[r-1] = -A[r-1,l-1]
			else:
				# If all rows are explored..
				if RH[n] <= 0:
					# Reduce matrix.
					A, CH, RH = hmreduce(A,CH,RH,LC,LR,SLC,SLR)

				# Re-start with first unexplored row.
				r = int(RH[n])                  
				# Get column of next free zero in row r.
				l = int(CH[r-1])                
				# Advance "column of next free zero".
				CH[r-1] = -A[r-1,l-1]
				# If this zero is last in the list..
				if A[r-1,l-1] == 0:
					# ...remove row r from unexplored list.
					RH[n] = RH[r-1]
					RH[r-1] = 0

			# While the column l is labelled, i.e. in path.
			while LC[l-1] != 0:
				# If row r is explored..
				if RH[r-1] == 0:
					# If all rows are explored..
					if RH[n] <= 0:
						# Reduce cost matrix.
						A, CH, RH = hmreduce(A,CH,RH,LC,LR,SLC,SLR)

					# Re-start with first unexplored row.
					r = int(RH[n])      

				# Get column of next free zero in row r.
				l = int(CH[r-1])    

				# Advance "column of next free zero".
				CH[r-1] = -A[r-1,l-1]

				# If this zero is last in list..
				if A[r-1,l-1] == 0:
					# ...remove row r from unexplored list.
					RH[n] = RH[r-1]
					RH[r-1] = 0

		 	# If the column found is unassigned..
			if C[l-1] == 0:
		 		# Flip all zeros along the path in LR,LC.
				A, C, U = hmflip(A,C,LC,LR,U,l,r)
				break
			else:
				# ...else add zero to path.
				# Label column l with row r.
				LC[l-1] = r

				# Add l to the set of labelled columns.
				SLC.append(l)

				# Continue with the row assigned to column l.
				r = int(C[l-1])   

				# Label row r with column l.
				LR[r-1] = l

				# Add r to the set of labelled rows.
				SLR.append(r)

	 # Calculate the total cost.
	temp_sparse = MATLAB_sparse(np.int64(C),np.array([x+1 for x in range(orig.shape[1])]),1)
	T = sum(LogiMul2d(orig, temp_sparse))
	return C, T



def hminired(A):
# %HMINIRED Initial reduction of cost matrix for the Hungarian method.
# %
# %B=assredin(A)
# %A - the unreduced cost matris.
# %B - the reduced cost matrix with linked zeros in each row.

# % v1.0  96-06-13. Niclas Borlin, niclas@cs.umu.se.
# % Translated by Yicheng Wang on 2021-10-26
	m, n = A.shape
	# Subtract column-minimum values from each column.
	colMin = (np.min(A, 0)).reshape(1,-1) # 行向量
	A = A - colMin[np.zeros([m],dtype = 'int32'),:]

	# Subtract row-minimum values from each row.减去行最小值
	rowMin = np.min(A, 1).reshape(-1,1) # 列向量
	A = A - rowMin[:,np.zeros([n],dtype = 'int32')]

	# Get positions of all zeros.
	i = np.where(A == 0)[0]
	j = np.where(A == 0)[1]

	# Extend A to give room for row zero list header column.后续一列0
	A = np.hstack((A,np.zeros([m,1],dtype = 'int32')))
	for k in range(n):
		# Get all column in this row.
		cols = np.array((LogiMul1d(j,k == i)))
		# Insert pointers in matrix.
		A = MATLAB_hungarian187(A,k,n,cols)
	return A

def hminiass(A):
# %HMINIASS Initial assignment of the Hungarian method.
# %
# %[B,C,U]=hminiass(A)
# %A - the reduced cost matrix.
# %B - the reduced cost matrix, with assigned zeros removed from lists.
# %C - a vector. C(J)=I means row I is assigned to column J,
# %              i.e. there is an assigned zero in position I,J.
# %U - a vector with a linked list of unassigned rows.

# % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
# % Translated by Yicheng Wang on 2021-10-26
	n, np1 = A.shape

	# Initalize return vectors.
	C = np.zeros([n],dtype='int32')
	U = np.zeros([n+1],dtype='int32')

	# Initialize last/next zero "pointers".
	LZ = np.zeros([n])
	NZ = np.zeros([n])

	for i in range(n): #i '+0'    j,lj '+1'
		# Set j to first unassigned zero in row i.
		lj = n + 1 
		j = int(-A[i,lj-1])

		# Repeat until we have no more zeros (j==0) or we find a zero
		# in an unassigned column (c(j)==0).

		while C[j-1] != 0: 
			# Advance lj and j in zero list. 
			lj = j 
			j = -A[i,lj-1] 

			# Stop if we hit end of list.
			if j == 0:
				break

		if j != 0:
			# We found a zero in an unassigned column
			# Assign row i to column j.
			C[j-1] = i+1

			# Remove A(i,j) from unassigned zero list.
			A[i, lj-1] = A[i, j-1] 
			
			# Update next/last unassigned zero pointers.
			NZ[i] = -A[i,j-1] 
			LZ[i] = lj        
			
			# Indicate A(i,j) is an assigned zero
			A[i,j-1] = 0
		else:
			# We found no zero in an unassigned column.
			# Check all zeros in this row.
			lj = n + 1   
			j = -A[i,lj-1] 

			# Check all zeros in this row for a suitable zero in another row.
			while j != 0:
				# Check the in the row assigned to this colum
				r = C[j-1] 

				# Pick up last/next pointers.
				lm = int(LZ[r-1])  
				m = int(NZ[r-1])    

				# Check all unchecked zeros in free list of this row.
				while m != 0:
					# Stop if we find an unassigned column.
					if C[m-1] == 0:
						break

					# Advance one step in list.
					lm = m
					m = -A[r-1, lm-1] 

				if m == 0:
					# We failed on row r. Continue with next zero on row i.
					lj = j 
					j = -A[i, lj-1]
				else:
					# We found a zero in an unassigned column.
					# Replace zero at (r,m) in unassigned list with zero at (r,j)
					A[r-1, lm-1] = -j  
					A[r-1,j-1] = A[r-1,m-1] 

					# Update last/next pointers in row r.
					NZ[r-1] = -A[r-1,m-1] 
					LZ[r-1] = j 

					# Mark A(r,m) as an assigned zero in the matrix . . .
					A[r-1,m-1] = 0

					# ...and in the assignment vector.
					C[m-1] = r

					# Remove A(i,j) from unassigned list.
					A[i,lj-1] = A[i,j-1]

					# Update last/next pointers in row r.
					NZ[i] = -A[i, j-1]
					LZ[i] = lj

					# Mark A(r,m) as an assigned zero in the matrix . . .
					A[i,j-1] = 0

					# ...and in the assignment vector.
					C[j-1] = i+1

					# Stop search.
					break

	# Create vector with list of unassigned rows.
	# Mark all rows have assignment.
	r = np.zeros([n])
	rows = C[C!=0]
	r[np.int64(rows)-1] = rows
	empty = np.where(r==0)[0]

	# Create vector with linked list of unassigned rows.
	U = np.zeros([n+1])
	U[Concat_Array322(n, empty)] = Concat_Array322(empty+1, 0) # DEBUG +1

	return A, C, U

def hmflip(A,C,LC,LR,U,l,r):
# %HMFLIP Flip assignment state of all zeros along a path.
# %
# %[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
# %Input:
# %A   - the cost matrix.
# %C   - the assignment vector.
# %LC  - the column label vector.
# %LR  - the row label vector.
# %U   - the 
# %r,l - position of last zero in path.
# %Output:
# %A   - updated cost matrix.
# %C   - updated assignment vector.
# %U   - updated unassigned row list vector.

# % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
# 	Translated by Yicheng Wang 2021-11-24
	n = A.shape[0]
	
	while 1:
		# Move assignment in column l to row r.
		C[l-1] = r

		# Find zero to be removed from zero list..

		# Find zero before this.
		m = np.where(A[r-1,:] == -l)[0]

		# Link past this zero.
		A[r-1,m-1] = A[r-1,l-1]

		A[r-1, l-1] = 0   ## ??? DEBUG A[r, l] = 0

		# If this was the first zero of the path..
		if LR[r-1] < 0:
			# remove row from unassigned row list and return.
			U[n] = U[r-1]
			U[r-1] = 0
			return A, C, U
		else:
			# Move back in this row along the path and get column of next zero.
			l = int(LR[r-1])

			# Insert zero at (r,l) first in zero list.
			A[r-1,l-1] = A[r-1,n]
			A[r-1,n] = -l

			# Continue back along the column to get row of next zero in path.
			r = int(LC[l-1])

	return A, C, U

def hmreduce(A,CH,RH,LC,LR,SLC,SLR):
# %HMREDUCE Reduce parts of cost matrix in the Hungerian method.
# %
# %[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
# %Input:
# %A   - Cost matrix.
# %CH  - vector of column of 'next zeros' in each row.
# %RH  - vector with list of unexplored rows.
# %LC  - column labels.
# %RC  - row labels.
# %SLC - set of column labels.
# %SLR - set of row labels.
# %
# %Output:
# %A   - Reduced cost matrix.
# %CH  - Updated vector of 'next zeros' in each row.
# %RH  - Updated vector of unexplored rows.

# % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
# 	Translated by Yicheng Wang 2021-11-24
	n = A.shape[0]

	# Find which rows are covered, i.e. unlabelled.
	coveredRows = LR == 0

	# Find which columns are covered, i.e. labelled.
	coveredCols = LC != 0

	r = np.where(coveredRows == False)[0]
	c = np.where(coveredCols == False)[0]

	# Get minimum of uncovered elements.
	m = np.min(np.min(Mat_hungarian415(A,r,c))) 

	# Subtract minimum from all uncovered elements.
	A = Mat_hungarian418(A,r,c,m,'subtract')

	# Check all uncovered columns..
	for j in c: # c中数值为python指标(0,1,2...

		for i in SLR:
			# If this is a (new) zero..
# 			print(A[i-1,j])
			if A[i-1,j] == 0:
				# If the row is not in unexplored list..
				if RH[i-1] == 0:
					# ...insert it first in unexplored list.
					RH[i-1] = RH[n]
					RH[n] = i
					# Mark this zero as "next free" in this row.
					CH[i-1] = j + 1 

				# Find last unassigned zero on row I.
				row = A[i-1,:]
				colsInList = -row[row<0]
				if len(colsInList) == 0:
					# No zeros in the list.
					l = n + 1
				else:
					l = LogiMul1d(colsInList, row[colsInList-1]==0)
				# Append this zero to end of list.
				A[i-1,l-1] = -(j + 1) 
	# Add minimum to all doubly covered elements.
	r = np.where(coveredRows)[0]
	c = np.where(coveredCols)[0]

	# Take care of the zeros we will remove.
	if len(np.where((Mat_hungarian415(A,r,c)<=0)))==1:
		i = np.where((Mat_hungarian415(A,r,c)<=0).reshape(-1,1))[0]
		j = np.where((Mat_hungarian415(A,r,c)<=0).reshape(-1,1))[1]
	elif len(np.where((Mat_hungarian415(A,r,c)<=0)))==2:
		i = np.where((Mat_hungarian415(A,r,c)<=0))[0]
		j = np.where((Mat_hungarian415(A,r,c)<=0))[1]
	i = r[i]
	j = c[j]

	for k in range(len(i)):
		# Find zero before this in this row.
		lj = np.where(A[i[k],:] == -(j[k]+1)) #??? debug +1

		# Link past it.
		A[i[k],lj] = A[i[k],j[k]]

		# Mark it as assigned.
		A[i[k],j[k]] = 0
	
	A = Mat_hungarian418(A,r,c,m,'add')
	
	return A, CH, RH