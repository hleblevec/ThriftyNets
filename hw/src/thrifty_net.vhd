library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity thrifty_net is
  port(
    clk : in std_logic;

    enable       : in  std_logic;
    valid_output : out std_logic;

    input_tensor   : in  t_data_tensor;
    classification : out t_fc_output

    );
end entity thrifty_net;

architecture rtl of thrifty_net is

  constant ITERATION_DELAY : natural := 5;

  signal fm_height  : natural range 1 to IN_FM_HEIGHT       := IN_FM_HEIGHT;
  signal fm_width   : natural range 1 to IN_FM_WIDTH        := IN_FM_WIDTH;
  signal iter_index : natural range 0 to NUM_ITERATIONS - 1 := 0;
  signal ds_enable  : std_logic;
  signal history    : t_history;

  signal delayer : std_logic_vector(ITERATION_DELAY - 1 downto 0);


begin

  ds_enable <= POOLING_STRATEGY(iter_index);

  valid_output_generator : process(clk)
  begin
    if rising_edge(clk) then


  thrifty_iteration : entity work.thrifty_iteration
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      iter_index    => iter_index,
      ds_enable     => ds_enable,
      history       => history,
      input_tensor  => input_tensor,
      output_tensor => output_tensor
      );

  global_pooling : entity work.global_pooling
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      input_tensor  => input_tensor,
      output_vector => fc_input
      );

  fc_layer : entity work.fc_layer
    port map (
      clk       => clk,
      fc_input  => fc_input,
      fc_output => classification
      );

end architecture rtl;
