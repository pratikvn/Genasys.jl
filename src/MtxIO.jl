using MatrixMarket
using SparseArrays

"""
    write(filename, matrix)

Writes a sparse matrix to a file in the matrix market format

# Examples
```julia-repl
mat = sprand(n, n)
julia>  write("mymatrix.mtx", mat)
```
"""
function write(fname::String, mat)
    MatrixMarket.mmwrite(fname, mat)
end


"""
    read(filename)

Reads a sparse matrix (in a the MatrixMarket format_header) from a file

# Examples
```julia-repl
julia>  mat = read("mymatrix.mtx")
```
"""
function read(fname::String)
    return MatrixMarket.mmread(fname)
end
