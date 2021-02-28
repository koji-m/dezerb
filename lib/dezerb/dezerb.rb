require 'set'
require 'numo/narray'

module Dezerb
  def using_config(name, value)
    old_value = Config.send(name)
    Config.send("#{name}=", value)
    begin
      yield
    ensure
      Config.send("#{name}=", old_value)
    end
  end

  def no_grad(&block)
    using_config('enable_backprop', false, &block)
  end

  def as_array(x)
    if x.instance_of? Variable or x.is_a? Numo::NArray
      x
    else
      Numo::NArray.cast(x)
    end
  end

  def as_variable(obj)
    if obj.instance_of? Variable
      obj
    else
      Variable.new(obj)
    end
  end

  module_function :using_config, :no_grad, :as_array, :as_variable

  class Config
    @@enable_backprop = true

    def self.enable_backprop()
      @@enable_backprop
    end

    def self.enable_backprop=(val)
      @@enable_backprop = val
    end
  end

  class Variable
    attr_accessor :data, :name, :grad, :creator, :generation

    def initialize(data, name = nil)
      unless data.nil?
        unless data.is_a? Numo::NArray
          raise TypeError, "#{data.class} is not supported"
        end
      end

      @data = data
      @name = name
      @grad = nil
      @creator = nil
      @generation = 0
    end

    def shape
      @data.shape
    end
    
    def ndim
      @data.ndim
    end

    def size
      @data.size
    end

    def length
      @data.length
    end

    def dtype
      @data.class
    end

    def inspect
      if @data.nil?
        'variable(nil)'
      else
        p = @data.inspect.gsub(/\n/, "\n" + " " * 9)
        "variable(#{p})"
      end
    end

    
    def +(rval)
      Add.new.(self, Dezerb.as_array(rval))
    end

    def *(rval)
      Mul.new.(self, Dezerb.as_array(rval))
    end

    def -(rval)
      Sub.new.(self, Dezerb.as_array(rval))
    end

    def /(rval)
      Div.new.(self, Dezerb.as_array(rval))
    end

    def -@
      Neg.new.(self)
    end

    def **(exp)
      Pow.new(exp).(self)
    end

    def coerce(other)
      [Variable.new(Dezerb.as_array(other)), self]
    end

    def set_creator(func)
      @creator = func
      @generation = func.generation + 1
    end

    def cleargrad()
      @grad = nil
    end

    def backward(retain_grad: false, create_graph: false)
      if @grad.nil?
        @grad = Variable.new(@data.new_ones)
      end

      funcs = []
      seen_set = Set.new
      add_func = Proc.new do |f|
        unless seen_set.include? f
          funcs << f
          seen_set << f
          funcs.sort_by! {|func| func.generation}
        end
      end

      add_func.(@creator)

      until funcs.empty?
        f = funcs.pop
        gys = f.outputs.map {|output| output.grad}

        Dezerb.using_config('enable_backprop', create_graph) do
          gxs = f.backward(*gys)
          unless gxs.instance_of? Array
            gxs = [gxs]
          end
          f.inputs.zip gxs do |x, gx|
            if x.grad.nil?
              x.grad = gx
            else
              x.grad = x.grad + gx
            end

            unless x.creator.nil?
              add_func.(x.creator)
            end
          end
        end

        unless retain_grad
          f.outputs.each {|y| y.grad = nil}
        end
      end
    end

    def reshape(*shape)
      if shape.length == 1 && shape[0].instance_of?(Array)
        shape = shape[0]
      end
      Functions.reshape self, shape
    end

    def transpose(*axes)
      if axes.length == 1 && axes[0].instance_of?(Array)
        axes = axes[0]
      end
      Functions.transpose self, axes
    end

    def T
      Functions.transpose self
    end

    def sum(axis: nil, keepdims: false)
      Functions.sum self, axis: axis, keepdims: keepdims
    end
  end

  
  class Function
    attr_accessor :inputs, :outputs, :generation

    def call(*inputs)
      inputs = inputs.map {|x| Dezerb.as_variable(x)}
      xs = inputs.map {|x| x.data}
      ys = forward(*xs)
      unless ys.instance_of? Array
        ys = [ys]
      end
      outputs = ys.map {|y| Variable.new(Dezerb.as_array(y))}
      
      if Config.enable_backprop
        @generation = inputs.max {|a, b| a.generation <=> b.generation}.generation
        outputs.each {|output| output.set_creator(self)}
        @inputs = inputs
        @outputs = outputs
      end

      outputs.length > 1 ? outputs : outputs[0]
    end

    

    def forward(in_data)
      raise NotImplementedError
    end

    def backward(gy)
      raise NotImplementedError
    end
  end

  class Add < Function
    def forward(x0, x1)
      @x0_shape, @x1_shape = x0.shape, x1.shape
      x0 + x1
    end

    def backward(gy)
      gx0, gx1 = gy, gy
      if @x0_shape != @x1_shape
        gx0 = Functions.sum_to gx0, @x0_shape
        gx1 = Functions.sum_to gx1, @x1_shape
      end
      [gx0, gx1]
    end
  end

  def add(x0, x1)
    Add.new.(x0, x1)
  end

  class Mul < Function
    def forward(x0, x1)
      x0 * x1
    end

    def backward(gy)
      x0, x1 = @inputs
      gx0, gx1 = gy * x1, gy * x0
      if x0.shape != x1.shape
        gx0 = Functions.sum_to(gx0, x0.shape)
        gx1 = Functions.sum_to(gx1, x1.shape)
      end
      [gx0, gx1]
    end
  end

  def mul(x0, x1)
    Mul.new.(x0, x1)
  end

  class Neg < Function
    def forward(x)
      -x
    end

    def backward(gy)
      -gy
    end
  end

  class Sub < Function
    def forward(x0, x1)
      @x0_shape, @x1_shape = x0.shape, x1.shape
      x0 - x1
    end

    def backward(gy)
      gx0, gx1 = gy, -gy
      if @x0_shape != @x1_shape
        gx0 = Functions.sum_to gx0, @x0_shape
        gx1 = Functions.sum_to gx1, @x1_shape
      end
      [gx0, gx1]
    end
  end

  class Div < Function
    def forward(x0, x1)
      x0 / x1
    end

    def backward(gy)
      x0, x1 = @inputs
      gx0, gx1 = gy / x1, gy * (-x0 / x1 ** 2)
      if x0.shape != x1.shape
        gx0 = Functions.sum_to gx0, x0.shape
        gx1 = Functions.sum_to gx1, x1.shape
      end
      [gx0, gx1]
    end
  end

  class Pow < Function
    attr_accessor :c
    def initialize(c)
      @c = c
    end

    def forward(x)
      x ** @c
    end

    def backward(gy)
      x = @inputs[0]
      c = @c
      c * x ** (c - 1) * gy
    end
  end
end

