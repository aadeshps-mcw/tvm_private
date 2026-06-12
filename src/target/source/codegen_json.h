#ifndef TVM_TARGET_CODEGEN_JSON_H_
#define TVM_TARGET_CODEGEN_JSON_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/function.h>
#include <tvm/runtime/module.h>
#include <sstream>
#include <string>

namespace tvm {
namespace codegen {

class CodeGenJSON : public tir::StmtExprVisitor {
 public:
  CodeGenJSON();

  std::string Build(const tir::PrimFunc& func);
  
  void VisitStmt_(const tir::EvaluateNode* op) override;
  void VisitStmt_(const tir::SeqStmtNode* op) override;
  void VisitStmt_(const tir::AssertStmtNode* op) override;
  void VisitStmt_(const tir::LetStmtNode* op) override;
  void VisitStmt_(const tir::AttrStmtNode* op) override;
  void VisitStmt_(const tir::ForNode* op) override;
  void VisitStmt_(const tir::WhileNode* op) override;
  void VisitStmt_(const tir::BufferStoreNode* op) override;
  void VisitStmt_(const tir::IfThenElseNode* op) override;
  void VisitStmt_(const tir::AllocateNode* op) override;
  void VisitStmt_(const tir::AllocateConstNode* op) override;

  void VisitExpr_(const tir::BufferLoadNode* op) override;

  void VisitExpr_(const tir::VarNode* op) override;
  void VisitExpr_(const tir::IntImmNode* op) override;
  void VisitExpr_(const tir::FloatImmNode* op) override;
  void VisitExpr_(const tir::StringImmNode* op) override;

  void VisitExpr_(const tir::CallNode* op) override;
  void VisitExpr_(const tir::CastNode* op) override;
  void VisitExpr_(const tir::SelectNode* op) override;
  void VisitExpr_(const tir::RampNode* op) override;
  void VisitExpr_(const tir::BroadcastNode* op) override;
  void VisitExpr_(const tir::ShuffleNode* op) override;

  void VisitExpr_(const tir::AndNode* op) override;
  void VisitExpr_(const tir::OrNode* op) override;
  void VisitExpr_(const tir::NotNode* op) override;
  
  void VisitExpr_(const tir::AddNode* op) override;
  void VisitExpr_(const tir::SubNode* op) override;
  void VisitExpr_(const tir::MulNode* op) override;
  void VisitExpr_(const tir::DivNode* op) override;
  void VisitExpr_(const tir::ModNode* op) override;
  void VisitExpr_(const tir::MinNode* op) override;
  void VisitExpr_(const tir::MaxNode* op) override;
  
  void VisitExpr_(const tir::EQNode* op) override;
  void VisitExpr_(const tir::NENode* op) override;
  void VisitExpr_(const tir::LTNode* op) override;
  void VisitExpr_(const tir::LENode* op) override;
  void VisitExpr_(const tir::GTNode* op) override;
  void VisitExpr_(const tir::GENode* op) override;

  

 private:
  std::ostringstream stream_;
  void PrintBinaryOp(const std::string& op_name, DataType dtype, const PrimExpr& a, const PrimExpr& b);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_CODEGEN_JSON_H_