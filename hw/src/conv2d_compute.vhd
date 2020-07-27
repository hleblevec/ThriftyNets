library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.fixed_pkg.all;

entity conv2d_compute is
  port(
    clk : in std_logic;

    enable     : in std_logic;
    reset_accu : in std_logic;

    --fm_height : in natural range 0 to IN_FM_HEIGHT - 1;
    fm_width : in natural range 0 to IN_FM_WIDTH + 1;

    output_channel_index : in natural range 0 to NUM_CHANNELS - 1;
    kernel_height_index  : in integer range KERNEL_LOW to KERNEL_HIGH;
    kernel_channel_index : in natural range 0 to NUM_CHANNELS - 1;

    -- weights : out t_kernel_matrix;

    input_row  : in  t_conv2d_feature_map_row;
    output_row : out t_conv2d_feature_map_row
    );
end entity conv2d_compute;

architecture rtl of conv2d_compute is

  type t_accu is array(0 to IN_FM_WIDTH + 1) of sfixed((DATA_INT_BW + WEIGHT_INT_BW + CONV2D_OVERFLOW) - 1 downto - (DATA_FRAC_BW + WEIGHT_FRAC_BW));
  signal accu : t_accu := (others => (others => '0'));

begin
  process(clk)
    variable accu_temp : t_accu := (others => (others => '0'));
  begin
    if rising_edge(clk) then
      if reset_accu = '1' then
        accu_temp := (others => (others => '0'));
        accu <= (others => (others => '0'));
        output_row <= (others => (others => '0'));
        for r in 0 to IN_FM_WIDTH + 1 loop
          if r < fm_width + 1 then
            output_row(r) <= resize(arg => accu(r), left_index => DATA_INT_BW - 1, right_index => -DATA_FRAC_BW, overflow_style => IEEE.fixed_float_types.fixed_saturate, round_style => IEEE.fixed_float_types.fixed_truncate);
            -- output_row(r) <= accu(r)(DATA_INT_BW - 1 downto -DATA_FRAC_BW);
          end if;
        end loop;
      elsif enable = '1' then
        -- weights <= KERNELS_LAYER(output_channel_index)(kernel_channel_index);
        for r in 1 to IN_FM_WIDTH loop
          for s2 in KERNEL_LOW to KERNEL_HIGH loop
            if r < fm_width + 1 then
              accu_temp(r) := resize(accu_temp(r) + input_row(r+s2)*KERNELS_LAYER(output_channel_index)(kernel_channel_index)(kernel_height_index + KERNEL_HIGH)(s2 + KERNEL_HIGH), DATA_INT_BW + WEIGHT_INT_BW + CONV2D_OVERFLOW - 1, -(DATA_FRAC_BW + WEIGHT_FRAC_BW));
            end if;
          end loop;
          accu(r) <= accu_temp(r);
        end loop;
      end if;
    end if;
  end process;

end architecture rtl;
