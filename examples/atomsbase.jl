using DFTK
using AtomsBase
using AtomIO
using Unitful
using UnitfulAtomic
using PyCall


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
# Use AtomIO package to read structure from file
#
system = load_system(joinpath(@__DIR__, "out.json"))
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

#
# DFTK constructor from data (deprecated syntax)
#
a = 10.26     # Silicon lattice constant (in Bohr)
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si    = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4.hgh"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
# kin_4 = Model(lattice; atoms, terms=[Kinetic()])
lda_4 = model_LDA(lattice, atoms)
