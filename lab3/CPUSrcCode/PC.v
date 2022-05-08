`timescale 1ns / 1ps
//  功能说明
    // 存储当前流水线需要执行的指令地址
// 输入
    // clk               时钟信号
    // NPC               NPC_Generator生成的下一条指令地址
// 输出
    // PC                流水线需要处理的指令地址
// 实验要求  
    // 无需修改

//DONE
module PC_IF(
    input wire clk, bubbleF, flushF,
    input wire [31:0] NPC,
    output reg [31:0] PC
    );

    initial PC = 0;
    
    always@(posedge clk)
        if (!bubbleF) 
        begin
            if (flushF)
                PC <= 0;
            else 
                PC <= NPC;
        end
    

endmodule