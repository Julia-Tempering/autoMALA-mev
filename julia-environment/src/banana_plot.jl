using Plots

x = range(-20, 20, length=100)
y = range(-200, 600, length=100)
f(x, y) = abs2(x)/20 + abs2(y-abs2(x))/(2*var_y) # p(x,y) = N(x;0,sqrt(10)^2)N(y;x^2,s^2)

using Statistics

b   = 5.0;
s_b = sqrt(inv(2*b));
scale = 2^6#2^-13
var_y = abs2(scale*s_b)
z = @. f(x', y)
levs = 10. .^ range(log10.(quantile(z,[0.1,0.9]))..., length=10)
contour(x, y, z,levels=levs)
plot!(abs2,x,linestyle=:dash,linewidth=2,label="x²")
savefig("banana.png")