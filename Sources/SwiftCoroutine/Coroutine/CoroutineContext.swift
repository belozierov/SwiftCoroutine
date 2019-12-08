//
//  CoroutineContext.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class CoroutineContext {
    
    public static let pool = Pool(creator: CoroutineContext.init)
    
    public typealias Block = () -> Void
    private let stack: UnsafeMutableRawBufferPointer
    private let returnPoint, resumePoint: UnsafeMutablePointer<Int32>
    
    @inline(__always)
    public init(stackSizeInPages: Int) {
        stack = .allocate(byteCount: stackSizeInPages * .pageSize, alignment: .pageSize)
        returnPoint = .allocate(capacity: .environmentSize)
        resumePoint = .allocate(capacity: .environmentSize)
    }
    
    @inline(__always)
    public convenience init() {
        self.init(stackSizeInPages: 32)
    }
    
    @discardableResult @inline(__always)
    open func start(block: @escaping Block) -> Bool {
        withUnsafePointer(to: { [unowned self] in
            block()
            longjmp(self.returnPoint, .finishFlag)
        }) {
            __start(returnPoint, stack.baseAddress?.advanced(by: stack.count), $0) {
                $0?.assumingMemoryBound(to: Block.self).pointee()
            }
        } == .finishFlag
     }
    
    @discardableResult @inline(__always)
    open func resume() -> Bool {
        __save(returnPoint, resumePoint) == .finishFlag
    }
    
    @inline(__always)
    open func suspend() {
        __save(resumePoint, returnPoint)
    }
    
    deinit {
        stack.deallocate()
        returnPoint.deallocate()
        resumePoint.deallocate()
    }
    
}

extension Int {
    
    fileprivate static let pageSize = sysconf(_SC_PAGESIZE)
    fileprivate static let environmentSize = MemoryLayout<jmp_buf>.size
    
}

extension Int32 {
    
    fileprivate static let finishFlag: Int32 = -1
    
}
