# need to install numo-gnuplot gem to run
require 'dezerb'
require 'numo/gnuplot'

include Dezerb
include Utils
F = Functions
L = Layers

x = Numo::DFloat.new(100, 1).rand
y = Numo::NMath.sin(2 * Math::PI * x) + Numo::DFloat.new(100, 1).rand

l1 = L::Linear.new(10)
l2 = L::Linear.new(1)

predict = Proc.new do |x|
  _y = l1.(x)
  _y = F.sigmoid _y
  l2.(_y)
end

lr = 0.2
iters = 10000

y_pred = nil
iters.times do |i|
  y_pred = predict.(x)
  loss = F.mean_squared_error y, y_pred

  l1.cleargrads
  l2.cleargrads
  loss.backward

  [l1, l2].each do |l|
    l.params.each do |param|
      param.data -= lr * param.grad.data
    end
  end
  
  p loss if i % 1000 == 0
end

Numo::gnuplot do
  set title: "neural network"
  sorted = x.to_a.zip(y_pred.data.to_a).sort_by {|v| v[0][0]}
  x_nar = Numo::NArray.cast(sorted.map {|v| v[0]})
  y_nar = Numo::NArray.cast(sorted.map {|v| v[1]})
  plot [x_nar, y_nar, w: "lines", t: "prediction"],
    [x, y, w: "points", t: "traindata"]
  pause mouse: %w[close]
end

