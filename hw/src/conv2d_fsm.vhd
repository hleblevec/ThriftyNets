library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.fixed_pkg.all;

entity conv2d_fsm is
  port(
    clk : in std_logic;

    data_available : in std_logic;

    enable_computation   : out std_logic;
    reset_accu           : out std_logic;
    -- output_channel_index : out natural range 0 to NUM_CHANNELS - 1;
    kernel_height_index  : out integer range KERNEL_LOW to KERNEL_HIGH;
    kernel_channel_index : out natural range 0 to NUM_CHANNELS - 1

    );
end entity conv2d_fsm;

architecture rtl of conv2d_fsm is

  type state_t is (wait_data, compute, reset);

  signal state                : state_t := wait_data;
  -- signal output_channel_count : natural range 0 to NUM_CHANNELS - 1;
  signal kernel_height_count  : integer range KERNEL_LOW to KERNEL_HIGH;
  signal kernel_channel_count : natural range 0 to NUM_CHANNELS - 1;


begin

  -- output_channel_index <= output_channel_count;
  kernel_height_index  <= kernel_height_count;
  kernel_channel_index <= kernel_channel_count;

  process(clk)
  begin
    if rising_edge(clk) then
      reset_accu         <= '0';
      enable_computation <= '0';
      case state is
        when wait_data =>
          -- output_channel_count <= 0;
          kernel_height_count  <= KERNEL_LOW;
          kernel_channel_count <= 0;

          if data_available = '1' then
            enable_computation <= '1';
            state <= compute;
          end if;

        when compute =>
          enable_computation <= '1';
          if kernel_channel_count < NUM_CHANNELS - 1 then
            if kernel_height_count < KERNEL_HIGH then
              kernel_height_count <= kernel_height_count + 1;
            else
              kernel_height_count  <= KERNEL_LOW;
              kernel_channel_count <= kernel_channel_count + 1;
            end if;
          elsif kernel_channel_count = NUM_CHANNELS - 1 then
            if kernel_height_count < KERNEL_HIGH then
              kernel_height_count <= kernel_height_count + 1;
            else
              kernel_height_count  <= KERNEL_LOW;
              kernel_channel_count <= 0;
              state                <= reset;
            end if;
          else
            kernel_channel_count <= 0;
            state                <= reset;
          end if;

        when reset =>
          reset_accu <= '1';
          state      <= wait_data;

      end case;
    end if;
  end process;

end architecture rtl;
