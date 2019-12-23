//
//  Coroutine+Thread.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension Coroutine {
    
    public static func current() throws -> Coroutine {
        if let coroutine = Thread.current.currentCoroutine { return coroutine }
        throw CoroutineError.mustBeCalledInsideCoroutine
    }
    
    public static var isInsideCoroutine: Bool {
        Thread.current.currentCoroutine != nil
    }
    
    public var isCurrent: Bool {
        self === Thread.current.currentCoroutine
    }
    
    func performAsCurrent(block: Block) {
        let thread = Thread.current
        let caller = thread.currentCoroutine
        thread.currentCoroutine = self
        block()
        thread.currentCoroutine = caller
    }
    
}

extension Thread {
    
    fileprivate var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
