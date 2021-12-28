# Convenience functions to make standard models


"""
Convenience constructor, which builds a standard atomic (kinetic + atomic potential) model.
Use `extra_terms` to add additional terms.
"""
function model_atomic(system::AbstractSystem; extra_terms=[], kwargs...)
    @assert !(:terms  in keys(kwargs))
    terms = [Kinetic(),
             AtomicLocal(),
             AtomicNonlocal(),
             Ewald(),
             PspCorrection(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] > 0
        terms = [terms..., Entropy()]
    end
    Model(system; model_name="atomic", terms=terms, kwargs...)
end


"""
Build a DFT model from the specified atoms, with the specified functionals.
"""
function model_DFT(system::AbstractSystem, xc::Xc; extra_terms=[], kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(xc.functionals, "+")
    model_atomic(system; extra_terms=[Hartree(), xc, extra_terms...],
                 model_name, kwargs...)
end
function model_DFT(system::AbstractSystem, functionals; kwargs...)
    model_DFT(system, Xc(functionals); kwargs...)
end


"""Build an LDA model (Teter93 parametrization)."""
model_LDA(system::AbstractSystem; kwargs...) = model_DFT(system, :lda_xc_teter93; kwargs...)


"""Build an PBE-GGA model."""
model_PBE(system::AbstractSystem; kwargs...) = model_DFT(system, [:gga_x_pbe, :gga_c_pbe]; kwargs...)


# TODO Temporary compatibility functions for old interface ...
function system_from_atoms(lattice::AbstractMatrix, atoms::AbstractVector)
    parsed_atoms = Atom[]
    for (element, positions) in atoms
        for pos in positions
            atom = Atom(chemical_symbol(element),
                        auconvert.(Ref(u"bohr"), lattice * pos),
                        potential=element)
            push!(parsed_atoms, atom)
        end
    end
    periodic_system(parsed_atoms, collect(eachcol(lattice)) * u"bohr")
end
model_atomic(lattice::AbstractMatrix, atoms::AbstractVector, args...; kwargs...) =
    model_atomic(system_from_atoms(lattice, atoms), args...; kwargs...)
model_DFT(lattice::AbstractMatrix, atoms::AbstractVector, args...; kwargs...) =
    model_DFT(system_from_atoms(lattice, atoms), args...; kwargs...)
model_LDA(lattice::AbstractMatrix, atoms::AbstractVector, args...; kwargs...) =
    model_LDA(system_from_atoms(lattice, atoms), args...; kwargs...)
model_PBE(lattice::AbstractMatrix, atoms::AbstractVector, args...; kwargs...) =
    model_PBE(system_from_atoms(lattice, atoms), args...; kwargs...)
