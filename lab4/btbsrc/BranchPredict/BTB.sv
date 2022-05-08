`timescale 1ns / 1ps
module BTB #(
    parameter SET_ADDR_LEN = 12
)(
    input clk, rst,
    input [31:0] pc_rd, pc_wr, pc_predict_wr, 
    input  br, write,
    output wire btb_hit, btb_br,
    output wire [31:0] pc_read_predict
);
localparam TAG_ADDR_LEN = 32 - SET_ADDR_LEN;
localparam BUFFER_SIZE = 1 << SET_ADDR_LEN;

wire [SET_ADDR_LEN - 1 : 0] read_addr, write_addr;
wire [TAG_ADDR_LEN - 1 : 0] read_tag, write_tag;

reg [TAG_ADDR_LEN - 1 : 0] tag_buffer [BUFFER_SIZE];
reg [32 - 1 : 0] buffer [BUFFER_SIZE];
reg valid [BUFFER_SIZE];
reg state [BUFFER_SIZE];

assign {read_tag, read_addr} = pc_rd;
assign {write_tag, write_addr} = pc_wr;

assign pc_read_predict = buffer[read_addr];
assign btb_hit = (tag_buffer[read_addr] == read_tag) && valid[read_addr];
assign btb_br = (tag_buffer[read_addr] == read_tag) && valid[read_addr] && state[read_addr]; 

integer i;
always @ (posedge clk or posedge rst) begin
    if (rst) begin
        for (i = 0; i < BUFFER_SIZE; i = i + 1) begin
            tag_buffer[i] <= 0;
            buffer[i] <= 0;
            valid[i] <= 0;
            state[i] <= 0;
        end
    end else begin
        if (write) begin
            if (br) begin
                tag_buffer[write_addr] <= write_tag;
                buffer[write_addr] <= pc_predict_wr;
                valid[write_addr] <= 1;
                state[write_addr] <= 1;
            end else begin
                tag_buffer[write_addr] <= write_tag;
                buffer[write_addr] <= pc_predict_wr;
                valid[write_addr] <= 1;
                state[write_addr] <= 0;
            end
        end
    end
end


endmodule