require 'set'

module Dezerb::Layers
  class Layer
    def initialize()
      @params = Set.new
    end

    def method_missing(name, *args)
      if m = name.match(/(\w+)=$/)
        var_name = m[1]
        Layer.class_eval do
          define_method name do |arg|
            @params << __method__[...-1]
            instance_variable_set("@#{var_name}", arg)
          end
        end
      else
        Layer.class_eval do
          define_method name do
            instance_variable_get("@#{name}")
          end
        end
      end
      send name, *args
    end

    def call(*inputs)
      outputs = forward(*inputs)
      unless outputs.instance_of? Array
        outputs = [outputs]
      end
      @inputs = inputs
      @outputs = outputs
      outputs.length > 1 ? outputs : outputs[0]
    end

    def forward(inputs)
      raise NotImplementedError
    end

    def params
      @params.map {|param_name| send param_name}
    end

    def cleargrads
      params.each do |param|
        param.cleargrad
      end
    end
  end

  class Linear < Layer
    def initialize(out_size, nobias: false, dtype: Numo::DFloat, in_size: nil)
      super()
      @in_size = in_size
      @out_size = out_size
      @dtype = dtype

      @w = Parameter.new(nil, name = 'w')
      _init_w unless @in_size.nil?

      if nobias
        @b = nil
      else
        @b = Parameter.new(dtype.zeros(out_size), name = 'b')
      end
      # ToDo
      @params << :w
      @params << :b
    end

    def _init_w
      i, o = @in_size, @out_size
      w_data = dtype.new(i, o).rand * Numo::NMath.sqrt(1.0 / i)
      @w.data = w_data
    end

    def forward(x)
      if @w.data.nil?
        @in_size = x.shape[1]
        _init_w
      end

      Dezerb::Functions.linear(x, @w, @b)
    end
  end
end
