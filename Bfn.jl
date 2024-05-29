
"""
    Bfn(u::T) where {T<:Real}
B-spline function.
"""
function Bfn(u::T) where {T<:Real}
    u = abs(u)
    if u <= one(T)
        value = one(T) - T(3)/T(2)*u^2 + T(3)/T(4)*u^3
    elseif u <= T(2)
        value = (T(2)-u)^3/T(4)
    else
        value = zero(T)
    end
    return value
end


"""
    dBfn(u::T) where {T<:Real}
Derivative with respect to u of B-spline function `Bfn(u)`.
"""
function dBfn(u::T) where {T<:Real}
    v = abs(u)
    s = sign(u)
    if v <= one(T)
        dBu = -T(3)*v + T(9)/T(4)*v^2
    elseif v <= T(2)
        dBu = -T(3)*(T(2)-v)^2/T(4)
    else
        dBu = zero(T)
    end
    return s*dBu
end