require 'numo/narray'

module Dezerb::Functions
  include Dezerb

  class Sin < Function
    def forward(x)
      Numo::NMath.sin x
    end

    def backward(gy)
      x, = @inputs
      gy * Functions.cos(x)
    end
  end

  def sin(x)
    Sin.new.(x)
  end

  class Cos < Function
    def forward(x)
      Numo::NMath.cos x
    end

    def backward(gy)
      x, = @inputs
      gy * -Functions.sin(x)
    end
  end

  def cos(x)
    Cos.new.(x)
  end

  class Tanh < Function
    def forward(x)
      Numo::NMath.tanh x
    end

    def backward(gy)
      y, = @outputs
      gy * (1 - y * y)
    end
  end

  def tanh(x)
    Tanh.new.(x)
  end

  class Reshape < Function
    def initialize(*shape)
      @shape = shape
    end

    def forward(x)
      @x_shape = x.shape
      x.reshape *@shape
    end

    def backward(gy)
      Functions.reshape gy, @x_shape
    end
  end

  def reshape(x, shape)
    x.shape == shape ? as_variable(x) : Reshape.new(*shape).(x)
  end
      
  class Transpose < Function
    def initialize(*axes)
      @axes = axes
    end

    def forward(x)
      x.transpose *@axes
    end

    def backward(gy)
      if @axes.empty?
        Functions.transpose gy
      else
        axes_len = @axes.length
        inv_axes = Numo::NArray.cast(@axes.map {|ax| ax % axes_len}).sort_index.to_a
        Functions.transpose gy, inv_axes
      end
    end
  end

  def transpose(x, axes = [])
    Transpose.new(*axes).(x)
  end

  class SumTo < Function
    def initialize(*shape)
      @shape = shape
    end

    def forward(x)
      @x_shape = x.shape
      x.sum_to *@shape
    end

    def backward(gy)
      Functions.broadcast_to(gy, @x_shape)
    end
  end

  def sum_to(x, shape)
    x.shape == shape ? as_variable(x) : SumTo.new(*shape).(x)
  end

  class BroadCastTo < Function
    def initialize(*shape)
      @shape = shape
    end

    def forward(x)
      @x_shape = x.shape
      x.broadcast_to *@shape
    end

    def backward(gy)
      Functions.sum_to gy, @x_shape
    end
  end

  def broadcast_to(x, shape)
    x.shape == shape ? as_variable(x) : BroadCastTo.new(*shape).(x)
  end

  class Sum < Function
    def initialize(axis, keepdims)
      @axis = axis
      @keepdims = keepdims
    end

    def forward(x)
      @x_shape = x.shape
      x.sum axis: @axis, keepdims: @keepdims
    end

    def backward(gy)
      gy = Utils.reshape_sum_backward gy, @x_shape, @axis, @keepdims
      Functions.broadcast_to gy, @x_shape
    end
  end

  def sum(x, axis: nil, keepdims: false)
    Sum.new(axis, keepdims).(x)
  end

  class MatMul < Function
    def forward(x, w)
      x.dot w
    end

    def backward(gy)
      x, w =  @inputs
      [Functions.matmul(gy, w.T), Functions.matmul(x.T, gy)]
    end
  end

  def matmul(x, w)
    MatMul.new.(x, w)
  end

  class MeanSquaredError < Function
    def forward(x0, x1)
      diff = x0 - x1
      (diff ** 2).sum / diff.length
    end

    def backward(gy)
      x0, x1 = @inputs
      diff = x0 - x1
      gx0 = gy * diff * (2.0 / diff.length)
      gx1 = -gx0
      [gx0, gx1]
    end
  end

  def mean_squared_error(x0, x1)
    MeanSquaredError.new.(x0, x1)
  end

  class Linear < Function
    def forward(x, w, b)
      y = x.dot w
      y += b unless b.nil?
      y
    end

    def backward(gy)
      x, w, b = @inputs
      gb = if b.data.nil? then nil else Functions.sum_to gy, b.shape end
      gx = Functions.matmul gy, w.T
      gw = Functions.matmul x.T, gy
      [gx, gw, gb]
    end
  end

  def linear(x, w, b = nil)
    Linear.new.(x, w, b)
  end

  class Sigmoid < Function
    def forward(x)
      1 / (1 + Numo::NMath.exp(-x))
    end

    def backward(gy)
      y = @outputs[0]
      gy * y * (1 - y)
    end
  end

  def sigmoid(x)
    Sigmoid.new.(x)
  end

  module_function(
    :sin, :cos, :tanh, :reshape, :transpose, :sum_to,
    :broadcast_to, :sum, :matmul, :mean_squared_error,
    :linear, :sigmoid
  )
end
