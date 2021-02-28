require 'tempfile'
require 'numo/narray'

class Numo::NArray
  def broadcast_to(*shape)
    from_shape = self.shape
    if shape.length < from_shape.length
      raise Exception.new('cannot shrink dimension')
    end

    (shape.length - from_shape.length).times {from_shape.unshift 1}

    reps = shape.zip(from_shape).map do |to, from|
      d, m = to.divmod from
      if d < 1
        raise Exception.new('cannot shrink size')
      end
      unless m == 0
        raise Exception.new('to-size must be a multiple of from-size')
      end
      d
    end
    self.reshape(*from_shape).tile(*reps)
  end

  def _squeeze_selected(ar, flgs)
    if flgs.length == 0
      ar
    elsif flgs[0] == true
      _squeeze_selected ar[0], flgs[1..]
    else
      ar.map {|a| _squeeze_selected a, flgs[1..]}
    end
  end

  def squeeze(axis)
    shape = self.shape
    ar = self.to_a
    flgs = [false] * shape.length
    if axis
      if axis.instance_of? Integer
        flgs[axis] = true
      elsif axis.instance_of? Array
        axis.each {|i| flgs[i] = true}
      else
        nil
      end
    else
      flgs = shape.map {|s| s == 1}
    end
    Numo::NArray.cast(_squeeze_selected(ar, flgs))
  end

  def sum_to(*shape)
    ndim = shape.length
    lead = self.ndim - ndim
    lead_axis = (0...lead).to_a

    axis = shape.select.with_index {|n, i| i if n == 1}
    y = self.sum *(lead_axis + axis), keepdims: true
    y = y.squeeze lead_axis if lead > 0

    y
  end
end

module Dezerb::Utils
  def _dot_var(v, verbose = false)
    dot_var = "%s [label=\"%s\", color=orange, style=filled]\n"

    name = v.name.nil? ? '' : v.name
    if verbose && !v.data.nil?
      unless v.name.nil?
        name += ': '
      end
      name += "#{v.shape} #{v.dtype}"
    end

    dot_var % [v.object_id, name]
  end

  def _dot_func(f)
    dot_func = "%s [label=\"%s\", color=lightblue, style=filled, shape=box]\n"
    ret = dot_func % [f.object_id, f.class.name.split('::').last]

    dot_edge = "%s -> %s\n"
    f.inputs.each {|x| ret += dot_edge % [x.object_id, f.object_id]}
    f.outputs.each {|y| ret += dot_edge % [f.object_id, y.object_id]}

    ret
  end
    
  def get_dot_graph(output, verbose = true)
    txt = ''
    funcs = []
    seen_set = Set.new

    add_func = Proc.new do |f|
      unless seen_set.include? f
        funcs << f
        seen_set << f
      end
    end

    add_func.(output.creator)
    txt += _dot_var(output, verbose)

    until funcs.empty?
      func = funcs.pop
      txt += _dot_func(func)
      func.inputs.each do |x|
        txt += _dot_var(x, verbose)

        unless x.creator.nil?
          add_func.(x.creator)
        end
      end
    end

    "digraph g {\n#{txt}}"
  end

  def plot_dot_graph(output, verbose: true, to_file: 'graph.png')
    dot_graph = get_dot_graph(output, verbose)

    Dir.mktmpdir do |dir|
      tf = Tempfile.open('graph', dir) do |f|
        f.puts dot_graph
        f
      end
      extension = File.extname(to_file)[1..-1]
      `dot #{tf.path} -T #{extension} -o #{to_file}`
    end
  end

  def reshape_sum_backward(gy, x_shape, axis, keepdims)
    ndim = x_shape.length
    arrayed_axis = axis
    if axis.nil?
      arrayed_axis = nil
    elsif !axis.instance_of? Array
      arrayed_axis = [axis]
    end

    unless ndim == 0 || arrayed_axis.nil? || keepdims
      actual_axis = arrayed_axis.map {|a| a >= 0 ? a : a + ndim}
      shape = gy.shape
      actual_axis.sort.each {|a| shape.insert a, 1}
    else
      shape = gy.shape
    end

    gy.reshape shape
  end
  
  module_function :_dot_var, :_dot_func, :get_dot_graph, :plot_dot_graph
end
