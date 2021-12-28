using DFTK
using AtomsBase
using Unitful
using UnitfulAtomic
using PyCall

# Super clunky ASE parser that should go to a separate package
function load_system(file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    ase_atoms = pyimport("ase.io").read(file)

    T = Float64
    cell_julia = convert(Array, ase_atoms.cell)  # Array of arrays
    box = [[T(cell_julia[j][i]) * u"Å" for i = 1:3] for j = 1:3]

    atoms = map(ase_atoms) do at
        atnum = convert(Int, at.number)
        Atom(atnum, at.position * u"Å"; magnetic_moment=at.magmom)
    end
    bcs = [p ? Periodic() : DirichletZero() for p in ase_atoms.pbc]
    atomic_system(atoms, box, bcs)
end


#
# AtomsBase from data
#
# (a) construct AtomsBase system
a = 10.26u"bohr"  # Silicon lattice constant
lattice = a / 2 * [[0, 1, 1.],  # Lattice as vector of vectors
                   [1, 0, 1.],
                   [1, 1, 0.]]
atoms  = [:Si => ones(3)/8, :Si => -ones(3)/8]
system = periodic_system(atoms, lattice; fractional=true)

# (b) Use inside DFTK
system = attach_psp(system; family="hgh", functional="lda")
kin_1  = Model(system; terms=[Kinetic()])  # low-level interface
lda_1  = model_LDA(system)                 # high-level interface

#
# AtomsBase from file
#
system = load_system("/tmp/out.json")
system = attach_psp(system; family="hgh", functional="lda")
kin_2  = Model(system; terms=[Kinetic()])  # low-level interface
lda_2  = model_LDA(system)                 # high-level interface

#
# DFTK constructor from data
#
a = 10.26     # Silicon lattice constant (in Bohr)
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si    = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4.hgh"))
atoms = [Si => ones(3)/8, Si => -ones(3)/8]
kin_3 = Model(lattice; atoms, terms=[Kinetic()])
lda_3 = model_LDA(lattice, atoms)
