# Convenience functions to make standard models


"""
Convenience constructor, which builds a standard atomic (kinetic + atomic potential) model.
Use `extra_terms` to add additional terms.
"""
function model_atomic(lattice::AbstractMatrix, atoms::Vector; extra_terms=[], kwargs...)
    @assert !(:terms in keys(kwargs))
    @assert !(:atoms in keys(kwargs))
    terms = [Kinetic(),
             AtomicLocal(),
             AtomicNonlocal(),
             Ewald(),
             PspCorrection(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    atoms = old_to_new(atoms)
    Model(lattice; model_name="atomic", atoms, terms, kwargs...)
end
function model_atomic(system::AbstractSystem; kwargs...)
    parsed = parse_system(system)
    model_atomic(parsed.lattice, parsed.atoms; parsed.kwargs..., kwargs...)
end


"""
Build a DFT model from the specified atoms, with the specified functionals.
"""
function model_DFT(lattice::AbstractMatrix, atoms::Vector, xc::Xc; extra_terms=[], kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(xc.functionals, "+")
    model_atomic(lattice, atoms; extra_terms=[Hartree(), xc, extra_terms...],
                 model_name, kwargs...)
end
function model_DFT(lattice::AbstractMatrix, atoms::Vector, functionals; kwargs...)
    model_DFT(lattice, atoms, Xc(functionals); kwargs...)
end
function model_DFT(system::AbstractSystem, args...; kwargs...)
    parsed = parse_system(system)
    model_DFT(parsed.lattice, parsed.atoms, args...; parsed.kwargs..., kwargs...)
end


"""
Build an LDA model (Teter93 parametrization) from the specified atoms.
"""
function model_LDA(lattice::AbstractMatrix, atoms::Vector; kwargs...)
    model_DFT(lattice, atoms, :lda_xc_teter93; kwargs...)
end
function model_LDA(system::AbstractSystem; kwargs...)
    model_DFT(system, :lda_xc_teter93; kwargs...)
end


"""
Build an PBE-GGA model from the specified atoms.
"""
function model_PBE(lattice::AbstractMatrix, atoms::Vector; kwargs...)
    model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe]; kwargs...)
end
function model_PBE(system::AbstractSystem; kwargs...)
    model_DFT(system, [:gga_x_pbe, :gga_c_pbe]; kwargs...)
end
