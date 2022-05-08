`timescale 1ns / 1ps
// 实验要求
    // 补全模块（阶段三）
// TODO
module CSR_Regfile(
    input wire clk,
    input wire rst,
    input wire CSR_write_en,
    input wire [11:0] CSR_write_addr,
    input wire [11:0] CSR_read_addr,
    input wire [31:0] CSR_data_write,
    output wire [31:0] CSR_data_read
    );
    
    // TODO: Complete this module

    /* FIXME: Write your code here... */
    reg [31:0] reg_file[4095:0];
    integer i;
    initial begin
        for(i = 0; i < 4096; i = i + 1) begin
            reg_file[i] = 32'b0;
        end
    end
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for (i = 1; i < 4096; i = i + 1) begin
                reg_file[i] <= 32'b0;
            end
        end else if (CSR_write_en) begin
            reg_file[CSR_write_addr] <= CSR_data_write;
        end
    end
    assign CSR_data_read = reg_file[CSR_read_addr];
endmodule
