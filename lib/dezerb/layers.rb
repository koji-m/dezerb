require 'set'

module Layers
  class Layer
    def initialize()
      @_params = Set.new
    end

    def method_missing(name, *args)
      if m = name.match(/(\w+)=$/)
        var_name = m[1]
        Layer.class_eval do
          define_method name do |arg|
            @_params << __method__[...-1]
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

    def cleargrads
      @_params.each do |param_name|
        send(param_name).cleargrad
      end
    end
  end
end
