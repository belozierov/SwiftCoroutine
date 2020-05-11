//
//  CoScopeTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoScopeTests: XCTestCase {
    
    private class TestCancellable: CoCancellable {
        
        private var callback: (() -> Void)?
        private(set) var isCanceled = false
        
        func whenComplete(_ callback: @escaping () -> Void) {
            self.callback = callback
        }
        
        func cancel() {
            callback?()
            callback = nil
            isCanceled = true
        }
        
        deinit {
            cancel()
        }
        
    }

    func testCancellable() {
       measure {
           let scope = CoScope()
           DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
               let item = TestCancellable()
               scope.add(item)
           }
           scope.cancel()
           DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
               let item = TestCancellable()
               scope.add(item)
            }
        }
    }
    
    func testConcurrency() {
        let scope = CoScope()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(50), execute: scope.cancel)
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            let item = TestCancellable().added(to: scope)
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(index % 100), execute: item.cancel)
        }
    }
    
    func testCancel() {
        let exp = expectation(description: "testCancel")
        exp.expectedFulfillmentCount = 2
        let scope = CoScope()
        let cancellable = TestCancellable()
        XCTAssertTrue(scope.isEmpty)
        scope.add(cancellable)
        XCTAssertFalse(scope.isEmpty)
        XCTAssertFalse(cancellable.isCanceled)
        cancellable.cancel()
        XCTAssertTrue(scope.isEmpty)
        scope.whenComplete { exp.fulfill() }
        scope.cancel()
        scope.whenComplete { exp.fulfill() }
        XCTAssertTrue(cancellable.isCanceled)
        XCTAssertTrue(scope.isEmpty)
        wait(for: [exp], timeout: 1)
    }
    
    func testDeinit() {
        var scope: CoScope! = CoScope()
        let cancellable = TestCancellable()
        scope.add(cancellable)
        XCTAssertFalse(cancellable.isCanceled)
        scope = nil
        XCTAssertTrue(cancellable.isCanceled)
    }
    
    

}
