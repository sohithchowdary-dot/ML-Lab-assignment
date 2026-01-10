def count_vowels_consonents(text):
    vowels_set =set("aeiouAEIOU")
    vowels =0
    consonents =0
    for ch in text:
        if ch.isalpha():
            if ch in vowels_set:
                vowels +=1
            else:
                consonents +=1
    return vowels,consonents

input_string ="Sohith Chowdary"
v,c =count_vowels_consonents(input_string)
print("vowels:",v,"consonents:",c)


def multiply_matrices(A,B):
    if len(A[0])!= lem(B):
        return "error cannot be multiplied"
    result= [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j]+= a[i][k]*B[k[j]]
    return result

A_rows= int(input("rows of matrix A:"))
A_cols= int(input("colums of matrix A: "))
B_rows= int(input("rows of matrix B: "))
B_cols= int(input("colums of matrix B: "))

print("enter matrix A:")
A= [list(map(int,input().split())) for _ in range(A_rows)]
print("enter matrix B:")
A= [list(map(int,input().split())) for _ in range(B_rows)]

print(multiply_matrices(A,B))
