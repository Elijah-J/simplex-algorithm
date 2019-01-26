import numpy as np
import math
import sys

# declare global variables to hold matrix and vector information
gMatrix = []
vectorb = []
vectorc = []

# Purpose: Write a function to read the data from the CSV
# Parameters: file name
# Output: structure
def dataread(filename):
    # stores data in structure depending on filename
    if filename == 'A.csv':
        structure = np.loadtxt(filename, delimiter=',', dtype=float)
    else:
        structure = np.loadtxt(filename, delimiter='\n', dtype=float)

    return structure

# Purpose: Print final dictionary
# Parameters: Basis array; Tableau
# Returns: Nothing
def pOneDictprint(basis, tableau):
    currSub = 0
    for x in range(0, len(basis)): # set up the basis (LHS)
        currSub = basis[x]
        print("x_", basis[x], " = ", sep="", end = "")
        print("%.6f" % tableau[x][-1], " ", end = "", sep = "")

        for y in range(0, (len(tableau[x]) - 2)): # set up the RHS
            if tableau[x][y] == 0: # takes care of x-values with 0-coefficients
                continue
            elif y == currSub: # takes care of basis variables on the RHS
                continue
            else:
                print("+ ", "%.6f" % -tableau[x][y], "x_", (y), " ", end = "", sep = "")
        print()

    print("z = ", "%.6f" % tableau[-1][-1], " ", sep = "", end = "")
    for z in range(0, (len(tableau[-1]) - 2)): # print the objective
        if tableau[-1][z] == 0:  # takes care of x-values with 0-coefficients
            continue
        print("+ ", "%.6f" % -tableau[-1][z], "x_", (z), " ", sep="", end="")

# Purpose: Print final dictionary repurposed for the first phase
# Parameters: Basis array; Tableau
# Returns: Nothing
def dictprint(basis, tableau):
    for x in range(0, len(basis)): # set up the basis (LHS)
        currSub = basis[x]
        print("x_", basis[x], " = ", sep="", end = "")
        print("%.6f" % tableau[x][-1], " ", end = "", sep = "")

        for y in range(0, (len(tableau[x]) - 2)): # set up the RHS
            if tableau[x][y] == 0: # takes care of x-values with 0-coefficients
                continue
            elif (y + 1) == currSub: # takes care of basis variables on the RHS
                continue
            else:
                print("+ ", "%.6f" % -tableau[x][y], "x_", (y + 1), " ", end = "", sep = "")
        print()

    print("z = ", "%.6f" % tableau[-1][-1], " ", sep = "", end = "")
    for z in range(0, (len(tableau[-1]) - 2)): # print the objective
        if tableau[-1][z] == 0:  # takes care of x-values with 0-coefficients
            continue
        print("+ ", "%.6f" % -tableau[-1][z], "x_", (z + 1), " ", sep="", end="")


# Purpose: Write a function to perform the simplex algorithm
# Parameters: Matrix and vectors
# Output: Dictionary result of iteration
def simplexiter(matrix, vectorb, vectorc):
    np.set_printoptions(suppress=True) # to ensure format printing

    objective = [-1 * x for x in vectorc] # sets up objective for the tableau

    # calculate number of slack variables
    slackCounter = 0
    for row in matrix:
        slackCounter = slackCounter + 1

    # calculate total number of variables
    totalCounter = 0
    for x in matrix[0]:
        totalCounter = totalCounter + 1
    totalCounter = totalCounter + slackCounter

    # calculate number of decision variables
    decisionCounter = totalCounter - slackCounter

    # account for slack values in objective function
    for x in range(0, slackCounter):
        objective.append(0)
    objective.append(1) # this accounts for the value of the objective function
    objective.append(0) # since we are setting our objective equal to 0

    # construct rows specifically for the tableau
    placeCounter = 0 # keeps track of our number of rows in our tableau
    tabRow = []
    for row in matrix:
        tabRow.append(row)
        tabRow[placeCounter] = tabRow[placeCounter].tolist()
        for x in range(0, slackCounter):
            if placeCounter == x:
                tabRow[placeCounter].append(1)
            else:
                tabRow[placeCounter].append(0)
        tabRow[placeCounter].append(0) # account for the value of z
        tabRow[placeCounter].append(vectorb[placeCounter]) # add solution
        placeCounter = placeCounter + 1

    tabRow.append(objective)
    tableau = tabRow # At this point, our initial tableau should be set up
    tempObjective = [0] + tableau[-1].copy()

    # set up our non-basis indexes
    nonBasis = []
    for x in range(1, (decisionCounter + 1)):
        nonBasis.append(x)

    # set up our basis indexes
    basis = []
    for x in range((decisionCounter + 1), (totalCounter + 1)):
        basis.append(x)

    # set up nonbasis and basis into one array
    collection = np.append(nonBasis, basis)

    # if any entry in our vector b is negative, enter the first phase
    if np.min(vectorb) < 0:

        # set our tableau up for our first phase
        newCol = []
        for x in range(0, len(tableau)):
            newCol.append(-1)
        newCol = np.array(newCol)

        # adds the -1 column to the front of our tableau
        tableau = np.concatenate((newCol[:, np.newaxis], tableau), axis=1)

        # create a new objective for the first phase
        newObjective = []

        for x in range(0, len(tableau[-1])):
            newObjective.append(0)
        newObjective[0] = 1
        newObjective[-2] = 1
        tableau[-1] = newObjective

        # update nonBasis to reflect our new x_0
        pOneNonBasis = [0]
        pOneNonBasis += nonBasis

        # perform the artificial first step
        enteringVarIndex = 0
        enteringVarSubscript = 0

        # get last index column
        lastIn = []
        for x in range(0, placeCounter):
            lastIn.append(tableau[x][-1])

        # get pivot column
        pivotColumn = []
        for x in range(0, placeCounter):
            pivotColumn.append(tableau[x][enteringVarIndex])

        # get leaving variable candidates
        leavingVarSuspects = np.divide(lastIn, pivotColumn)

        # get leaving variable
        try:
            leavingVar = np.max([x for x in leavingVarSuspects if x > 0]) # finds our leaving variable
        except ValueError:
            pass

        # find leaving variable index
        for i, x in enumerate(leavingVarSuspects):
            if x == leavingVar:
                leavingVarIndex = i;
                break

        # find leaving variable's subscript
        leavingVarSubscript = basis[leavingVarIndex]

        # isolate the row we are working with from the tableau (to get our leaving variable to 1)
        workingRow = tableau[leavingVarIndex]

        # get the row set up for row operations
        workingRow = [x / workingRow[enteringVarIndex] for x in workingRow]

        # modify tableau with modified index
        tableau[leavingVarIndex] = workingRow

        # for each row, reduce the value in the pivot column to 0
        counter = 0
        for row in tableau:
            if (counter == leavingVarIndex):
                counter = counter + 1
                continue
            mulVar = -1 * tableau[counter][enteringVarIndex]
            tableau[counter] = rowOp(mulVar, tableau[leavingVarIndex], tableau[counter])
            counter = counter + 1

        # update nonbasis
        for x in range(0, len(nonBasis)):
            if nonBasis[x] == enteringVarSubscript:
                nonBasis[x] = leavingVarSubscript
                break

        # update basis
        for x in range(0, len(basis)):
            if basis[x] == leavingVarSubscript:
                basis[x] = enteringVarSubscript
                break

        ########## NOTICE ##############
        # We have just done the first step of the first phase. We now loop through our tableau as normal
        # we get a an objective of the form 1 ... 1 0
        targetObjective = newObjective

        iterating = True
        iterationTotal = 1
        while (iterating):
            if np.array_equal(tableau[-1], targetObjective):
                print()
                print("We hath iterated throughout!")

                # remove x_0 column from tableau
                tableau = np.delete(tableau, 0, axis=1)
                tableau[-1] = np.delete(tempObjective, 0, axis=0)

                # repurpose objective function to only have nonbasis variables on RHS
                index = 1
                for x in range(0, len(tableau[-1] - 2)):
                    if index in basis and tableau[-1][index - 1] != 0:
                        opIndex = basis.index(index)
                        mulVar = -1*tableau[-1][index -1]
                        tableau[-1] = rowOp(mulVar, tableau[opIndex], tableau[-1])
                        index += 1
                    else:
                        index += 1

                break

            tempVal = tableau[-1][-1]
            tableau[-1][-1] = 0

            if np.min(tableau[-1]) >= 0 and (0 in basis):
                print()
                print("This LP Problem is Infeasible!")
                print()
                tableau[-1][-1] = tempVal
                pOneDictprint(basis, tableau)
                exit(0)

            tableau[-1][-1] = tempVal


            enteringVar = np.min(tableau[-1])  # store our entering variable
            alteredObjective = tableau[-1]
            tempVal = tableau[-1][-1]
            alteredObjective[-1] = 0

            enteringVarIndex = np.argmin(alteredObjective)  # store our entering variable's index
            enteringVarSubscript = enteringVarIndex
            tableau[-1][-1] = tempVal

            # collect last indices from tableau
            lastIn = []
            for x in range(0, placeCounter):
                lastIn.append(tableau[x][-1])

            # set up list of each value in pivot column
            pivotColumn = []
            for x in range(0, placeCounter):
                pivotColumn.append(tableau[x][enteringVarIndex])

            # takes care of any zeroes so we don't run into divByZeroErr
            for i, x in enumerate(pivotColumn):
                if x == 0:
                    pivotColumn[i] = np.finfo(np.double).tiny

            leavingVarSuspects = np.divide(lastIn, pivotColumn)  # gives us a list of possible candidates for our
                                                                # leaving variable

            # check if unbounded
            unbounded = True
            for x in leavingVarSuspects:
                if x > 0:
                    unbounded = False
                    break

            # If we are unbounded, end iteration
            if unbounded == True:
                print("This dataset is unbounded!")
                print(iterationTotal)
                exit(0)

            try:
                leavingVar = np.min([x for x in leavingVarSuspects if x > 0])  # finds our leaving variable
            except ValueError:
                pass

            # find leaving varaible index
            for i, x in enumerate(leavingVarSuspects):
                if x == leavingVar:
                    leavingVarIndex = i;
                    break

            # find leaving variable's subscript
            leavingVarSubscript = basis[leavingVarIndex]

            # isolate the row we are working with from the tableau (to get our leaving variable to 1)
            workingRow = tableau[leavingVarIndex]

            # for modifying temp objective
            objectiveDiv = workingRow[enteringVarIndex]

            # get the row set up for row operations
            workingRow = [x / workingRow[enteringVarIndex] for x in workingRow]

            # modify tableau with modified index
            tableau[leavingVarIndex] = workingRow

            # for each row, reduce the value in the pivot column to 0
            counter = 0
            for row in tableau:
                if (counter == leavingVarIndex):
                    counter = counter + 1
                    continue
                mulVar = -1 * tableau[counter][enteringVarIndex]
                tableau[counter] = rowOp(mulVar, tableau[leavingVarIndex], tableau[counter])
                counter = counter + 1

            # update nonbasis
            for x in range(0, len(nonBasis)):
                if nonBasis[x] == enteringVarSubscript:
                    nonBasis[x] = leavingVarSubscript
                    break

            # update basis
            for x in range(0, len(basis)):
                if basis[x] == leavingVarSubscript:
                    basis[x] = enteringVarSubscript
                    break

            iterationTotal += 1

            # detect an infeasible set of data
            if iterationTotal == 50:
                print("This dataset is infeasible!")
                dictprint(basis, tableau)
                break

    iterating = True
    iterationTotal = 0
    while(iterating):
        enteringVar = np.min(tableau[-1]) # store our entering variable
        enteringVarIndex = np.argmin(tableau[-1]) # store our entering variable's index
        enteringVarSubscript = enteringVarIndex + 1

        # collect last indices from tableau
        lastIn = []
        for x in range(0, placeCounter):
            lastIn.append(tableau[x][-1])

        # set up list of each value in pivot column
        pivotColumn = []
        for x in range(0, placeCounter):
            pivotColumn.append(tableau[x][enteringVarIndex])

        # takes care of any zeroes so we don't run into divByZeroErr
        for i, x in enumerate(pivotColumn):
            if x == 0:
                pivotColumn[i] = np.finfo(np.double).tiny

        leavingVarSuspects = np.divide(lastIn, pivotColumn) # gives us a list of possible candidates for our
                                                            # leaving variable

        # check if unbounded
        unbounded = True
        for x in leavingVarSuspects:
            if x > 0:
                unbounded = False

        # If we are unbounded, end iteration
        if unbounded == True:
            print(iterationTotal)
            print("This dataset is unbounded!")
            break

        try:
            leavingVar = np.min([x for x in leavingVarSuspects if x > 0]) # finds our leaving variable
        except ValueError:
            pass

        # find leaving varaible index
        for i, x in enumerate(leavingVarSuspects):
            if x == leavingVar:
                leavingVarIndex = i;
                break;

        # find leaving variable's subscript
        leavingVarSubscript = basis[leavingVarIndex]


        # isolate the row we are working with from the tableau (to get our leaving variable to 1)
        workingRow = tableau[leavingVarIndex]

        # get the row set up for row operations
        workingRow = [x/workingRow[enteringVarIndex] for x in workingRow]

        # modify tableau with modified index
        tableau[leavingVarIndex] = workingRow

        # for each row, reduce the value in the pivot column to 0
        counter = 0
        for row in tableau:
            if (counter == leavingVarIndex):
                counter = counter + 1
                continue
            mulVar = -1 * tableau[counter][enteringVarIndex]
            tableau[counter] = rowOp(mulVar, tableau[leavingVarIndex], tableau[counter])
            counter = counter + 1

        # update nonbasis
        for x in range(0, len(nonBasis)):
            if nonBasis[x] == enteringVarSubscript:
                nonBasis[x] = leavingVarSubscript
                break

        # update basis
        for x in range(0, len(basis)):
            if basis[x] == leavingVarSubscript:
                basis[x] = enteringVarSubscript
                break

        # update collection
        collection = np.append(nonBasis, basis)

        # check objective and break if there are no values less than 1
        finished = True
        for x in tableau[-1]:
            if x < 0:
                finished = False
                break

        # detects if we have finished
        if finished == True:
            iterating = False

        # detect an infeasible dictionary
        iterationTotal += 1

        # detect an infeasible set of data
        if iterationTotal == 50:
            print("This dataset is infeasible!")
            dictprint(basis, tableau)
            break

    print("This solution is optimal!")
    dictprint(basis, tableau)

# Purpose: performs a row opearation involving addition to a row multiplied by a scalar
# Parameters: multiplier, row added from, row added to
# Returns: New row
def rowOp(multiplier, addFrom, addTo):
    counter = 0
    for x in addFrom:
        addTo[counter] = addTo[counter] + (multiplier * x)
        counter = counter + 1

    return addTo

# main method
def main():
    matrix = dataread('A.csv')
    vectorb = dataread('b.csv')
    vectorc = dataread('c.csv')

    simplexiter(matrix, vectorb, vectorc)

if __name__ == '__main__':
    main()
