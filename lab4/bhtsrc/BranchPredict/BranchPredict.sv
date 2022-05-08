`timescale 1ns / 1ps
module BranchPredict # (
    parameter SET_ADDR_LEN = 12
)(
    input clk, rst,
    input [31:0] PC_rd_IF, PC_wr_EX, PC_br_target, 
    input  br, write,
    output reg br_predict_miss,
    output reg [31:0] NPC
);
wire btb_hit, btb_br, bht_br;
wire [32 - 1 : 0 ] PC_rd_IF_4;
assign PC_rd_IF_4 = PC_rd_IF + 4;

wire [32 - 1 : 0] btb_pc_predict;
reg [32 - 1 : 0] PC_rd_ID, PC_rd_EX;
wire br_pred_IF;
reg br_pred_ID, br_pred_EX;

assign br_pred_IF = btb_hit & bht_br;


always @(posedge clk or posedge rst) begin
    if (rst) begin
        PC_rd_ID <= 32'b0;
        PC_rd_EX <= 32'b0;
        br_pred_ID <= 0;
        br_pred_EX <= 0;
    end else begin
        PC_rd_ID <= PC_rd_IF_4;
        PC_rd_EX <= PC_rd_ID;
        br_pred_ID <= br_pred_IF;
        br_pred_EX <= br_pred_ID;
    end
end

always @ (*) begin
    if (write) begin
        if (br_pred_EX == br) begin
            NPC = br_pred_IF ? btb_pc_predict : PC_rd_IF_4;
            br_predict_miss = 0;
        end else begin
            NPC = br ? PC_br_target : PC_rd_EX;
            br_predict_miss = 1;
        end
    end else begin
        NPC = br_pred_IF ? btb_pc_predict : PC_rd_IF_4;
        br_predict_miss = 0;
    end
end

BTB #(
    .SET_ADDR_LEN(SET_ADDR_LEN)
    ) BTB1 (
        .clk(clk),
        .rst(rst),
        .pc_rd(PC_rd_IF),
        .pc_wr(PC_wr_EX),
        .pc_predict_wr(PC_br_target),
        .br(br),
        .write(write),
        .btb_hit(btb_hit),
        .btb_br(btb_br),
        .pc_read_predict(btb_pc_predict)
    );

BHT #(
    .SET_ADDR_LEN(SET_ADDR_LEN)
    ) BHT1 (
        .clk(clk),
        .rst(rst),
        .pc_rd(PC_rd_IF),
        .pc_wr(PC_wr_EX),
        .br(br),
        .write(write),
        .bht_br(bht_br)
    );

endmodule