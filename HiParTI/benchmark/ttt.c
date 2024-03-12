/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ParTI.h>
#include <assert.h>
#include <string.h>

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -f config file\n");
    printf("\n");
}

/*
* tensor expression
* e ((X * Y) * Z)
* n pair contraction
* m x y
* m x y
* t 
*/

struct ContractInfo {
    int m;
    int x[8];
    int y[8];
};

struct ContractInfo* parse_config(char *config_file, char *expression, int *n, int *t){
    FILE* file = fopen(config_file, "r"); 

    if (file == NULL) {
        perror("File opening failed");
        return 1;
    }
    struct ContractInfo* res;

    char buffer[1024]; // Buffer to store each line
    bool is_expression = false;
    int index = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        printf("\n%s", buffer);
        if (!is_expression) {
            strcpy(expression, buffer);
            is_expression = true;
        }
        if (buffer[0] == 'n') {
            char* token = strtok(buffer, " ");
            while (token != NULL) {
                token = strtok(NULL, " ");
                if (token == NULL) {
                    continue;;
                }
                *n = atoi(token);
                printf("n : %d\n", *n);
                res = (struct ContractInfo*)malloc(*n * sizeof(struct ContractInfo));
            }
        }
        if (buffer[0] == 'm') {
            bool is_x = false;
            bool is_y = false;
            bool is_m = false;
            char* token = strtok(buffer, " ");
            while (token != NULL) {
                token = strtok(NULL, " ");
                if (token == NULL) {
                    continue;
                }
                if (!is_m){
                    int num_dim = atoi(token);
                    res[index].m = num_dim;
                    is_m = true;
                    printf("m : %d\n", res[index].m);
                }
                else if (!is_x){
                    if (strcmp(token, "x") == 0){
                        continue;
                    }
                    for (int i = 0; i < res[index].m; i++) {
                        res[index].x[i] = atoi(token);
                        printf("x : %d\n", res[index].x[i]);
                        token = strtok(NULL, " ");
                        if (token == NULL) {
                            continue;
                        }
                    }
                    is_x = true;
                }
                else if (!is_y){
                   if (strcmp(token, "y") == 0){
                        continue;
                    }
                    for (int i = 0; i < res[index].m; i++) {
                        res[index].y[i] = atoi(token);
                        printf("y : %d\n", res[index].y[i]);   
                        token = strtok(NULL, " ");
                        if (token == NULL) {
                            continue;;
                        }
                    }
                    is_y = true;
                }

            }
            index++;
        }
        if (buffer[0] == 't') {
            char* token = strtok(buffer, " ");
            while (token != NULL) {
                token = strtok(NULL, " ");
                if (token == NULL) {
                    continue;;
                }
                *t = atoi(token);
                //printf("t : %d\n", *t);
            }
        }
    }

    fclose(file); // Close the file when done
    return res;
}

// Stack structure for operators
typedef struct {
    char *data;
    int top;
    int capacity;
} OperatorStack;

// Stack structure for operands
typedef struct {
    sptSparseTensor *data;
    int top;
    int capacity;
} OperandStack;

// Initialize operator stack
OperatorStack* createOperatorStack(int capacity) {
    OperatorStack *stack = (OperatorStack*)malloc(sizeof(OperatorStack));
    stack->data = (char*)malloc(capacity * sizeof(char));
    stack->top = -1;
    stack->capacity = capacity;
    return stack;
}

// Push an operator onto the operator stack
void pushOperator(OperatorStack *stack, char operator) {
    stack->data[++stack->top] = operator;
}

// Pop an operator from the operator stack
char popOperator(OperatorStack *stack) {
    return stack->data[stack->top--];
}

// Get the top operator from the operator stack
char topOperator(OperatorStack *stack) {
    return stack->data[stack->top];
}

// Initialize operand stack
OperandStack* createOperandStack(int capacity) {
    OperandStack *stack = (OperandStack*)malloc(sizeof(OperandStack));
    stack->data = (sptSparseTensor*)malloc(capacity * sizeof(sptSparseTensor));
    stack->top = -1;
    stack->capacity = capacity;
    return stack;
}

// Push an operand onto the operand stack
void pushOperand(OperandStack *stack, sptSparseTensor operand) {
    stack->data[++stack->top] = operand;
}

// Pop an operand from the operand stack
sptSparseTensor popOperand(OperandStack *stack) {
    return stack->data[stack->top--];
}

// Check if the character is an operator
int isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/');
}

// Get the precedence of an operator
int precedence(char op) {
    if (op == '+' || op == '-')
        return 1;
    if (op == '*' || op == '/')
        return 2;
    return 0;
}

typedef struct {
    char name;
    char* filename;
} Variable;

//Evaluate the expression with parentheses
sptSparseTensor evaluateExpression(const char *expression, Variable* variables, int numVariables, struct ContractInfo* info, int nt, int output_sorting, int placement) {
    int len = strlen(expression);
    OperatorStack *operatorStack = createOperatorStack(len);
    OperandStack *operandStack = createOperandStack(len);

    sptSparseTensor X, Y, Z;
    int binarize_order = 0;
    for (int i = 0; i < len; i++) {
        if (expression[i] == '(') {
            pushOperator(operatorStack, expression[i]);
        } else if (expression[i] == ')') {
            while (topOperator(operatorStack) != '(') {
                sptSparseTensor operand2 = popOperand(operandStack);
                sptSparseTensor operand1 = popOperand(operandStack);
                char operator = popOperator(operatorStack);
                printf("operand1: \n");
                sptSparseTensorStatus(&operand1, stdout);
                printf("operand2: \n");
                sptSparseTensorStatus(&operand2, stdout);   
                
                printf("Original Tensors: \n"); 
                // sptAssert(sptDumpSparseTensor(&operand1, 0, stdout) == 0);
                // sptAssert(sptDumpSparseTensor(&operand2, 0, stdout) == 0);   

                // contract
                sptIndex * cmodes_X = NULL, * cmodes_Y = NULL;
                sptIndex num_cmodes = info[binarize_order].m;
                cmodes_X = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
                cmodes_Y = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
                sptAssert(cmodes_X != NULL && cmodes_Y != NULL);
                printf("cmode m %d\n", info[binarize_order].m);
                for(sptIndex i = 0; i < num_cmodes; ++ i) {
                    cmodes_X[i] = info[binarize_order].x[i];
                    cmodes_Y[i] = info[binarize_order].y[i];
                    printf("x y %d %d\n", cmodes_X[i], cmodes_Y[i]);
                }
                
                sptAssert(sptSparseTensorMulTensor(&Z, &operand1, &operand2, num_cmodes, cmodes_X, cmodes_Y, nt, output_sorting, placement) == 0);
                binarize_order++;
                pushOperand(operandStack, Z);
            }
            popOperator(operatorStack); // Pop the '('
        } 
        else if (expression[i] == '*'){
            pushOperator(operatorStack, expression[i]);
        }
        else if (isalpha(expression[i])) {
            char variableName = expression[i];
            char* variableFilename; // Default value for undefined variables
            for (int j = 0; j < numVariables; j++) {
                if (variables[j].name == variableName) {
                    variableFilename = variables[j].filename;
                    break;
                }
            }
            if (strlen(variableFilename) == 0) {
                printf("Variable %c is undefined.\n", variableName);
                exit(-1);
            }
            sptAssert(sptLoadSparseTensor(&X, 1, variableFilename) == 0);
            pushOperand(operandStack, X);
        }
    }

    sptSparseTensor result = popOperand(operandStack);
    free(operatorStack->data);
    free(operatorStack);
    free(operandStack->data);
    free(operandStack);
    return result;
}


int main(int argc, char *argv[]) {
    char expression[1024];
    int n;
    int t;
    char *config_file = argv[1];
    printf("config file: %s\n", config_file);
    struct ContractInfo* info = parse_config(config_file, &expression, &n, &t);
    printf("expression: %s\n", expression);
    printf("n: %d\n", n);
    printf("t: %d\n", t);

    int input_file_num = atoi(argv[2]);
    printf("input_file_num: %d\n", input_file_num);

    Variable variables[input_file_num];

    for (int i = 0; i < input_file_num * 2; i += 2){
        // printf("var name: %s\n", argv[3+i]);
        // printf("var input file: %s\n", argv[3+i + 1]);
        Variable newvar;
        newvar.name = argv[3+i][0];
        newvar.filename = argv[3+i + 1];

        variables[i/2] = newvar;
    }


    // Variable variables[] = {
    //     { 'X', "/home/yufan/data_3cent_small_sparta/Int_scipy.tns" },
    //     { 'Y', "/home/yufan/data_3cent_small_sparta/C_scipy.tns" },
    //     { 'Z', "/home/yufan/data_3cent_small_sparta/Phat_scipy.tns" }
    // };
    int numVariables = input_file_num; // sizeof(variables) / sizeof(variables[0]);
    
    for (int i = 0; i < numVariables; i++){
        printf("var name: %c\n", variables[i].name);
        printf("var input file: %s\n", variables[i].filename);
    }


    // for (int i = 0; i < n; i++){
    //     printf("m %d\n", info[i].m);
    //     for (int j = 0; j < info[i].m; j++){
    //         printf("x %d\n", info[i].x[j]);
    //     }   
    //     for (int j = 0; j < info[i].m; j++){
    //         printf("y %d\n", info[i].y[j]);
    //     }   
    // }

    sptSparseTensor result = evaluateExpression(expression, variables, numVariables, info, t, 1, 0);
    sptSparseTensorStatus(&result, stdout);
    // sptAssert(sptDumpSparseTensor(&result, 0, stdout) == 0);

    return 0;
}



