`timescale 1ns / 1ps
module BHT #(
    parameter SET_ADDR_LEN = 12
)(
    input clk, rst,
    input [31:0] pc_rd, pc_wr,
    input br, write,
    output wire bht_br
);
localparam BUFFER_SIZE = 1 << SET_ADDR_LEN;
localparam THRSHOLD = 1;

reg [1 : 0] STATE [BUFFER_SIZE];
wire [SET_ADDR_LEN - 1 : 0] read_addr, write_addr;

assign read_addr = pc_rd[SET_ADDR_LEN - 1 : 0];
assign write_addr = pc_wr[SET_ADDR_LEN - 1 : 0];
assign bht_br = (STATE[read_addr] > THRSHOLD) ? 1 : 0;

integer i;
always @ (posedge clk or posedge rst) begin
    if (rst) begin
        for (i = 0; i < BUFFER_SIZE; i = i + 1) begin
            STATE[i] <= 2'b0;
        end
    end else if (write) begin
        if (br) begin
            STATE[write_addr] <= (STATE[write_addr] == 2'b11) ? 2'b11 : STATE[write_addr]+1;
        end else begin
            STATE[write_addr] <= (STATE[write_addr] == 2'b00) ? 2'b00 : STATE[write_addr]-1;
        end
    end
end

endmodule