# # Polarizability using Zygote

using DFTK
using LinearAlgebra
using ForwardDiff
using Zygote

## Construct PlaneWaveBasis given a particular electric field strength
## Again we take the example of a Helium atom.
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He]  # Helium at the center of the box
positions = [[1/2; 1/2; 1/2]]
function make_basis(ε::T; a=10., Ecut=5) where T  # too small Ecut, only for efficient debugging
    lattice=T(a) * Mat3(I(3))  # lattice is a cube of ``a`` Bohrs
    # model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];
    #                   extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
    #                   symmetries=false)

    # TODO: make missing terms differentiable
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection(),
        Entropy(),
        Hartree(),
        ExternalFromReal(r -> -ε * (r[1] - a/2)),
        Xc([:lda_x, :lda_c_vwn])
    ]
    model = Model(lattice, atoms, positions; terms, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    # @assert isdiag(basis.model.lattice)
    a  = basis.model.lattice[1, 1]
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    sum(rr .* ρ) * basis.dvol
end

## Function to compute the dipole for a given field strength
function compute_dipole(ε; tol=1e-8, kwargs...)
    scfres = self_consistent_field(make_basis(ε; kwargs...), tol=tol)
    dipole(scfres.basis, scfres.ρ)
end


f = compute_dipole(0.0)

polarizability_fd = let
    ε = 0.001
    (compute_dipole(ε) - f) / ε
end

g_forward = ForwardDiff.derivative(compute_dipole, 0.0)

g_reverse = Zygote.gradient(compute_dipole, 0.0)


println("f: ", f, " fd: ", polarizability_fd, " AD (forward): ", g_forward, " AD (reverse): ", g_reverse)

