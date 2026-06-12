#include "codegen_json.h"
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/target/codegen.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/data_type.h>

#include <fstream>

namespace tvm {
namespace codegen {

class JSONModuleNode : public runtime::ModuleNode {
 public:
  explicit JSONModuleNode(std::string code) : code_(code) {}

  const char* type_key() const override { return "json"; }


  void SaveToFile(const String& file_name, const String& format) override {
    std::string fmt = format;
    
    std::string fname_str = file_name; 
    if (fmt.empty() && fname_str.find(".json") != std::string::npos) {
      fmt = "json";
    }
    
    ICHECK_EQ(fmt, "json") << "Can only save to .json format";
    
    std::ofstream out(file_name);
    ICHECK(out.is_open()) << "Cannot open file " << file_name;
    out << code_;
    out.close();
  }

  runtime::PackedFunc GetFunction(const String& name,
                                  const ObjectPtr<Object>& sptr_to_self) override {
    return nullptr;
  }

 private:
  std::string code_;
};

CodeGenJSON::CodeGenJSON() {}

std::string CodeGenJSON::Build(const tir::PrimFunc& func) {
  stream_ << "{\n";
  stream_ << "  \"function_name\": \"" 
          << func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("default_func") 
          << "\",\n";
  stream_ << "  \"body\": ";
  
  this->VisitStmt(func->body);
  
  stream_ << "\n}";
  return stream_.str();
}

void CodeGenJSON::VisitStmt_(const tir::EvaluateNode* op) {
  stream_ << "{\n";
  stream_ << "  \"type\": \"Evaluate\",\n";
  stream_ << "  \"expr\": ";
  this->VisitExpr(op->value);
  stream_ << "\n}";
}

void CodeGenJSON::VisitExpr_(const tir::VarNode* op) {
  stream_ << "{\"type\": \"Var\", \"name\": \"" << op->name_hint 
          << "\", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) << "\"}";
}

void CodeGenJSON::VisitExpr_(const tir::IntImmNode* op) {
  stream_ << "{\"type\": \"IntImm\", \"value\": " << op->value 
          << ", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) << "\"}";
}

void CodeGenJSON::VisitExpr_(const tir::FloatImmNode* op) {
  stream_ << "{\"type\": \"FloatImm\", \"value\": " << op->value 
          << ", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) << "\"}";
}

void CodeGenJSON::PrintBinaryOp(const std::string& op_name, DataType dtype, const PrimExpr& a, const PrimExpr& b) {
  stream_ << "{\"type\": \"" << op_name 
          << "\", \"dtype\": \"" << runtime::DLDataType2String(dtype) 
          << "\", \"a\": ";
  this->VisitExpr(a);
  stream_ << ", \"b\": ";
  this->VisitExpr(b);
  stream_ << "}";
}

void CodeGenJSON::VisitStmt_(const tir::AssertStmtNode* op) {
  stream_ << "{\n  \"type\": \"Assert\",\n  \"condition\": ";
  this->VisitExpr(op->condition);
  stream_ << ",\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::LetStmtNode* op) {
  stream_ << "{\n  \"type\": \"Let\",\n  \"var\": ";
  this->VisitExpr(op->var);
  stream_ << ",\n  \"value\": ";
  this->VisitExpr(op->value);
  stream_ << ",\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::AttrStmtNode* op) {
  stream_ << "{\n  \"type\": \"AttrStmt\",\n  \"attr_key\": \"" << op->attr_key 
          << "\",\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::SeqStmtNode* op) {
  stream_ << "[\n";
  for (size_t i = 0; i < op->seq.size(); ++i) {
    this->VisitStmt(op->seq[i]);
    if (i < op->seq.size() - 1) stream_ << ",\n";
  }
  stream_ << "\n]";
}

void CodeGenJSON::VisitStmt_(const tir::ForNode* op) {
  stream_ << "{\n  \"type\": \"For\",\n  \"loop_var\": ";
  this->VisitExpr(op->loop_var);
  stream_ << ",\n  \"min\": ";
  this->VisitExpr(op->min);
  stream_ << ",\n  \"extent\": ";
  this->VisitExpr(op->extent);
  stream_ << ",\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::BufferStoreNode* op) {
  stream_ << "{\n  \"type\": \"BufferStore\",\n  \"buffer\": \"" << op->buffer->name << "\"";
  stream_ << ",\n  \"value\": ";
  this->VisitExpr(op->value);
  stream_ << ",\n  \"indices\": [\n    ";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    this->VisitExpr(op->indices[i]);
    if (i < op->indices.size() - 1) stream_ << ",\n    ";
  }
  stream_ << "\n  ]\n}";
}

void CodeGenJSON::VisitExpr_(const tir::BufferLoadNode* op) {
  stream_ << "{\"type\": \"BufferLoad\", \"buffer\": \"" << op->buffer->name 
          << "\", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\", \"indices\": [";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    this->VisitExpr(op->indices[i]);
    if (i < op->indices.size() - 1) stream_ << ", ";
  }
  stream_ << "]}";
}

void CodeGenJSON::VisitStmt_(const tir::IfThenElseNode* op) {
  stream_ << "{\n  \"type\": \"IfThenElse\",\n  \"condition\": ";
  this->VisitExpr(op->condition);
  stream_ << ",\n  \"then_case\": ";
  this->VisitStmt(op->then_case);
  
  if (op->else_case.defined()) {
    stream_ << ",\n  \"else_case\": ";
    this->VisitStmt(op->else_case.value());
  }
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::AllocateNode* op) {
  stream_ << "{\n  \"type\": \"Allocate\",\n  \"buffer_var\": \"" << op->buffer_var->name_hint 
          << "\",\n  \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\",\n  \"extents\": [\n    ";
  for (size_t i = 0; i < op->extents.size(); ++i) {
    this->VisitExpr(op->extents[i]);
    if (i < op->extents.size() - 1) stream_ << ",\n    ";
  }
  stream_ << "\n  ],\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitExpr_(const tir::CallNode* op) {
  stream_ << "{\n  \"type\": \"Call\",\n  \"op\": \"";
  
  if (const auto* op_ptr = op->op.as<OpNode>()) {
    stream_ << op_ptr->name;
  } else if (const auto* gv_ptr = op->op.as<GlobalVarNode>()) {
    stream_ << gv_ptr->name_hint;
  } else {
    stream_ << "unknown_call";
  }
  
  stream_ << "\",\n  \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\",\n  \"args\": [\n    ";
          
  for (size_t i = 0; i < op->args.size(); ++i) {
    this->VisitExpr(op->args[i]);
    if (i < op->args.size() - 1) stream_ << ",\n    ";
  }
  stream_ << "\n  ]\n}";
}

void CodeGenJSON::VisitExpr_(const tir::CastNode* op) {
  stream_ << "{\"type\": \"Cast\", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\", \"value\": ";
  this->VisitExpr(op->value);
  stream_ << "}";
}

void CodeGenJSON::VisitExpr_(const tir::SelectNode* op) {
  stream_ << "{\n  \"type\": \"Select\",\n  \"condition\": ";
  this->VisitExpr(op->condition);
  stream_ << ",\n  \"true_value\": ";
  this->VisitExpr(op->true_value);
  stream_ << ",\n  \"false_value\": ";
  this->VisitExpr(op->false_value);
  stream_ << "\n}";
}


void CodeGenJSON::VisitStmt_(const tir::WhileNode* op) {
  stream_ << "{\n  \"type\": \"While\",\n  \"condition\": ";
  this->VisitExpr(op->condition);
  stream_ << ",\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitStmt_(const tir::AllocateConstNode* op) {
  stream_ << "{\n  \"type\": \"AllocateConst\",\n  \"buffer_var\": \"" << op->buffer_var->name_hint 
          << "\",\n  \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\",\n  \"extents\": [\n    ";
  for (size_t i = 0; i < op->extents.size(); ++i) {
    this->VisitExpr(op->extents[i]);
    if (i < op->extents.size() - 1) stream_ << ",\n    ";
  }
  stream_ << "\n  ],\n  \"body\": ";
  this->VisitStmt(op->body);
  stream_ << "\n}";
}

void CodeGenJSON::VisitExpr_(const tir::RampNode* op) {
  // Represents a vector like [base, base + stride, base + 2*stride, ...]
  stream_ << "{\"type\": \"Ramp\", \"base\": ";
  this->VisitExpr(op->base);
  stream_ << ", \"stride\": ";
  this->VisitExpr(op->stride);
  
  stream_ << ", \"lanes\": " << op->lanes << "}";
}

void CodeGenJSON::VisitExpr_(const tir::BroadcastNode* op) {
  stream_ << "{\"type\": \"Broadcast\", \"value\": ";
  this->VisitExpr(op->value);
  stream_ << ", \"lanes\": " << op->lanes << "}";
}

void CodeGenJSON::VisitExpr_(const tir::ShuffleNode* op) {
  stream_ << "{\n  \"type\": \"Shuffle\",\n  \"vectors\": [\n    ";
  
  for (size_t i = 0; i < op->vectors.size(); ++i) {
    this->VisitExpr(op->vectors[i]);
    if (i < op->vectors.size() - 1) stream_ << ",\n    ";
  }
  
  stream_ << "\n  ],\n  \"indices\": [\n    ";
  
  for (size_t i = 0; i < op->indices.size(); ++i) {
    this->VisitExpr(op->indices[i]);
    if (i < op->indices.size() - 1) stream_ << ",\n    ";
  }
  
  stream_ << "\n  ]\n}";
}

void CodeGenJSON::VisitExpr_(const tir::StringImmNode* op) {
  stream_ << "{\"type\": \"StringImm\", \"value\": \"" << op->value << "\"}";
}

void CodeGenJSON::VisitExpr_(const tir::AndNode* op) { PrintBinaryOp("And", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::OrNode* op) { PrintBinaryOp("Or", op->dtype, op->a, op->b); }

void CodeGenJSON::VisitExpr_(const tir::NotNode* op) {
  stream_ << "{\"type\": \"Not\", \"dtype\": \"" << runtime::DLDataType2String(op->dtype) 
          << "\", \"a\": ";
  this->VisitExpr(op->a);
  stream_ << "}";
}

void CodeGenJSON::VisitExpr_(const tir::AddNode* op) { PrintBinaryOp("Add", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::SubNode* op) { PrintBinaryOp("Sub", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::MulNode* op) { PrintBinaryOp("Mul", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::DivNode* op) { PrintBinaryOp("Div", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::ModNode* op) { PrintBinaryOp("Mod", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::MinNode* op) { PrintBinaryOp("Min", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::MaxNode* op) { PrintBinaryOp("Max", op->dtype, op->a, op->b); }

void CodeGenJSON::VisitExpr_(const tir::EQNode* op) { PrintBinaryOp("EQ", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::NENode* op) { PrintBinaryOp("NE", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::LTNode* op) { PrintBinaryOp("LT", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::LENode* op) { PrintBinaryOp("LE", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::GTNode* op) { PrintBinaryOp("GT", op->dtype, op->a, op->b); }
void CodeGenJSON::VisitExpr_(const tir::GENode* op) { PrintBinaryOp("GE", op->dtype, op->a, op->b); }


TVM_REGISTER_GLOBAL("target.build.json")
.set_body_typed([](IRModule mod, Target target) -> runtime::Module {
    
    std::ostringstream full_json;
    full_json << "{\n  \"modules\": [\n";

    bool first = true;
    for (auto kv : mod->functions) {
      if (kv.second->IsInstance<tir::PrimFuncNode>()) {
        if (!first) full_json << ",\n";
        
        CodeGenJSON cg;
        
        tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(kv.second);
        full_json << cg.Build(prim_func);
        
        first = false;
      }
    }
    full_json << "\n  ]\n}";

    return runtime::Module(make_object<JSONModuleNode>(full_json.str()));
});

}  // namespace codegen
}  // namespace tvm