using Genasys
using Documenter

DocMeta.setdocmeta!(Genasys, :DocTestSetup, :(using Genasys); recursive=true)

makedocs(;
    modules=[Genasys],
    authors="Pratik Nayak",
    sitename="Genasys.jl",
    format=Documenter.HTML(;
        canonical="https://pratikvn.github.io/Genasys.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pratikvn/Genasys.jl",
    devbranch="main",
)
